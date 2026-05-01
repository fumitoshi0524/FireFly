import torch
import importlib
import warnings


_NEXT_HANDLE = 1
_REGISTERED_HANDLES: set[int] = set()
_BIT_GRAD_CACHE: dict[int, torch.Tensor] = {}
_FF_EXT = False
_EXT_CANDIDATES = ("firefly_int8_ext", "firefly_bitnet_ext")
_EXT_WARNED = False


def next_bit_handle() -> int:
    global _NEXT_HANDLE
    while _NEXT_HANDLE in _REGISTERED_HANDLES:
        _NEXT_HANDLE += 1
    handle = _NEXT_HANDLE
    _REGISTERED_HANDLES.add(handle)
    _NEXT_HANDLE += 1
    return handle


def register_bit_handle(handle: int) -> int:
    global _NEXT_HANDLE
    handle = int(handle)
    if handle <= 0:
        raise ValueError(f"handle must be > 0, got {handle}")
    _REGISTERED_HANDLES.add(handle)
    _BIT_GRAD_CACHE.pop(handle, None)
    if handle >= _NEXT_HANDLE:
        _NEXT_HANDLE = handle + 1
    return handle


def release_bit_handle(handle: int) -> None:
    handle = int(handle)
    _REGISTERED_HANDLES.discard(handle)
    _BIT_GRAD_CACHE.pop(handle, None)


def consume_bit_grad(handle: int) -> torch.Tensor | None:
    return _BIT_GRAD_CACHE.pop(int(handle), None)


def _get_firefly_ext():
    global _FF_EXT, _EXT_WARNED
    if _FF_EXT is not False:
        return _FF_EXT
    for mod_name in _EXT_CANDIDATES:
        try:
            _FF_EXT = importlib.import_module(mod_name)
            break
        except Exception:
            _FF_EXT = None
    if _FF_EXT is None and not _EXT_WARNED:
        _EXT_WARNED = True
        warnings.warn(
            "firefly_int8_ext is unavailable; falling back to Python INT8 kernels.",
            RuntimeWarning,
            stacklevel=2,
        )
    return _FF_EXT


def quantize_fp_to_int8(weight: torch.Tensor, eps: float = 1e-8):
    if weight.ndim != 2:
        raise ValueError(
            f"weight must be 2D [out_features, in_features], got {tuple(weight.shape)}"
        )
    w = weight.float()
    scale = w.abs().amax(dim=1, keepdim=True).clamp_min(float(eps)) / 127.0
    q = torch.round(w / scale).clamp(-127, 127).to(torch.int8)
    return q.contiguous(), scale.squeeze(1).contiguous()


def _quantize_activation_per_token(x2d: torch.Tensor, eps: float = 1e-8):
    x = x2d.float()
    scale = x.abs().amax(dim=1, keepdim=True).clamp_min(float(eps)) / 127.0
    q = torch.round(x / scale).clamp(-127, 127).to(torch.int8)
    return q.contiguous(), scale.contiguous()


def _quantize_per_column(x2d: torch.Tensor, eps: float = 1e-8):
    """Per-column (per output-channel) quantisation for bitsandbytes outer-product dequant."""
    x = x2d.float()
    scale = x.abs().amax(dim=0, keepdim=True).clamp_min(float(eps)) / 127.0
    q = torch.round(x / scale).clamp(-127, 127).to(torch.int8)
    return q.contiguous(), scale.contiguous()  # q: [N, C]  scale: [1, C]


def _col_stats_from_row_quant(q_i8: torch.Tensor, eps: float = 1e-8):
    """Estimate per-column max-abs from a row-wise quantised int8 tensor."""
    return q_i8.float().abs().amax(dim=0).clamp_min(float(eps)) / 127.0  # [ncol]


class Int8LinearFn(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x2d: torch.Tensor,
        int_weight: torch.Tensor,
        weight_scale: torch.Tensor,
        bias: torch.Tensor | None,
        handle: int,
    ):
        x_saved = x2d.contiguous().float()
        x_q, x_scale = _quantize_activation_per_token(x_saved)
        # Pure int8 matmul → int32 output.  No weight dequant.
        out_i32 = x_q.to(torch.int32).matmul(int_weight.to(torch.int32).t())
        # Dequant output only for the next layer (RMSNorm / attention need float).
        out = out_i32.float() * (x_scale * weight_scale.float().view(1, -1))
        if bias is not None:
            out.add_(bias.float().view(1, -1))

        # Save only integer tensors + float scales.
        ctx.save_for_backward(
            x_q, x_scale, int_weight, weight_scale, out_i32,
        )
        ctx.handle = int(handle)
        ctx.input_dtype = x2d.dtype
        ctx.has_bias = bias is not None
        return out.to(dtype=x2d.dtype)

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        x_q, x_scale, int_weight, weight_scale, out_i32 = ctx.saved_tensors
        grad_out_f32 = grad_out.contiguous().float()
        ws_f32 = weight_scale.float()

        # -- grad_in (backward input) -------------------------------------------
        #   grad_in = (grad_out · weight_scale) @ int_weight
        #   Per-token quantise grad_out, int8 matmul, dequant only the result.
        go_q, go_scale = _quantize_activation_per_token(
            grad_out_f32 * ws_f32.view(1, -1)
        )
        grad_in_i32 = go_q.to(torch.int32).matmul(int_weight.to(torch.int32))
        grad_in = grad_in_i32.float() * go_scale

        # -- grad_w (int8 weight gradient) --------------------------------------
        #   bitsandbytes-style: per-OUTPUT-channel quantise grad_out,
        #   per-INPUT-channel stats from x_q, outer-product dequant.
        go_pc_q, go_pc_scale = _quantize_per_column(
            grad_out_f32 * ws_f32.view(1, -1)
        )
        x_col_scale = _col_stats_from_row_quant(x_q)
        grad_w_i32 = go_pc_q.t().to(torch.int32).matmul(x_q.to(torch.int32))
        grad_w = grad_w_i32.float() * go_pc_scale.t() * x_col_scale.view(1, -1)
        cached = _BIT_GRAD_CACHE.get(ctx.handle)
        if cached is None:
            _BIT_GRAD_CACHE[ctx.handle] = grad_w.to(dtype=torch.bfloat16)
        else:
            cached.add_(grad_w.to(dtype=torch.bfloat16))

        # -- grad_weight_scale (now a trainable Parameter) -----------------------
        #   out = out_i32 · x_scale · weight_scale
        #   ∂out/∂weight_scale = out_i32 · x_scale
        grad_weight_scale = (
            grad_out_f32 * out_i32.float() * x_scale
        ).sum(dim=0).to(dtype=weight_scale.dtype)
        grad_bias = (
            grad_out_f32.sum(dim=0).to(dtype=grad_out.dtype) if ctx.has_bias else None
        )
        return (
            grad_in.to(dtype=ctx.input_dtype),
            None,              # int_weight — buffer, no gradient
            grad_weight_scale, # weight_scale — now a trainable Parameter
            grad_bias,
            None,              # handle
        )


@torch.no_grad()
def update_int8_weight_(int_weight: torch.Tensor, delta_q: torch.Tensor) -> None:
    if int_weight.dtype != torch.int8:
        raise TypeError(f"int_weight must be torch.int8, got {int_weight.dtype}")
    if delta_q.shape != int_weight.shape:
        raise ValueError(
            f"delta_q shape mismatch: expected {tuple(int_weight.shape)}, got {tuple(delta_q.shape)}"
        )
    ext = _get_firefly_ext()
    if ext is not None and hasattr(ext, "ff_int8_weight_update_"):
        ext.ff_int8_weight_update_(
            int_weight.contiguous(), delta_q.to(torch.int32).contiguous()
        )
        return
    if int_weight.is_cuda:
        raise RuntimeError(
            "INT8 CUDA optimizer kernel is required but not loaded. "
            "Build and load firefly_bitnet_ext (ff_int8_weight_update_)."
        )
    next_weight = int_weight.to(torch.int16).add_(delta_q.to(torch.int16))
    next_weight.clamp_(-127, 127)
    int_weight.copy_(next_weight.to(torch.int8))
