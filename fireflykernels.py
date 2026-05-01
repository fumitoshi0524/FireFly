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


# ---------------------------------------------------------------------------
#  bitsandbytes-style:  INT8 weight storage  +  BF16 Tensor-Core compute
#
#  int_weight (int8) is the only source of truth — no shadow weights.
#  Dequantised to bf16 on-the-fly; all three matmuls (fwd, grad_in, grad_w)
#  run on BF16 Tensor Cores via torch.matmul (cuBLAS).
#
#  Memory:  x2d (bf16) + int_weight (int8) saved for backward.
#           w_bf16 is recomputed in backward from int_weight + weight_scale
#           (int8→bf16 conversion is cheap, saves ~1.1 GB cached bf16 weights).
# ---------------------------------------------------------------------------

class Int8LinearFn(torch.autograd.Function):

    @staticmethod
    def forward(
        ctx,
        x2d: torch.Tensor,          # [N, K]  bf16  (from autocast)
        int_weight: torch.Tensor,   # [O, K]  int8  (buffer, no grad)
        weight_scale: torch.Tensor, # [O]     float (trainable Parameter)
        bias: torch.Tensor | None,  # [O]     float
        handle: int,
    ):
        # Dequant int8 → bf16  (transient compute cache, not a shadow weight)
        w_bf16 = int_weight.to(torch.bfloat16) * weight_scale.to(torch.bfloat16).unsqueeze(1)

        # Forward: BF16 Tensor-Core matmul via cuBLAS
        out = torch.matmul(x2d, w_bf16.t())
        if bias is not None:
            out.add_(bias)

        ctx.save_for_backward(x2d, int_weight, weight_scale)
        ctx.handle = int(handle)
        ctx.has_bias = bias is not None
        return out

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        x2d, int_weight, weight_scale = ctx.saved_tensors
        grad_out = grad_out.contiguous()

        # Recompute bf16 weight (cheap: int8→bf16 convert + element-wise scale)
        w_bf16 = int_weight.to(torch.bfloat16) * weight_scale.to(torch.bfloat16).unsqueeze(1)

        # grad_in  = grad_out @ W          BF16 Tensor Cores
        grad_in = torch.matmul(grad_out, w_bf16)

        # grad_w   = grad_out^T @ x2d      BF16 Tensor Cores
        grad_w = torch.matmul(grad_out.t(), x2d)
        cached = _BIT_GRAD_CACHE.get(ctx.handle)
        if cached is None:
            _BIT_GRAD_CACHE[ctx.handle] = grad_w.to(dtype=torch.bfloat16)
        else:
            cached.add_(grad_w.to(dtype=torch.bfloat16))

        # grad_weight_scale  =  (grad_w ⊙ int_weight) · 1_row
        #   ∂W_bf16/∂weight_scale[i] = int_weight[i,:]
        #   ∂loss/∂weight_scale[i]   = Σ_j grad_w[i,j] * int_weight[i,j]
        grad_weight_scale = (
            grad_w.float() * int_weight.float()
        ).sum(dim=1).to(dtype=weight_scale.dtype)

        grad_bias = grad_out.sum(dim=0).to(dtype=grad_out.dtype) if ctx.has_bias else None

        return (
            grad_in,            # x2d
            None,               # int_weight (buffer, gradient via cache)
            grad_weight_scale,  # weight_scale (trainable Parameter)
            grad_bias,          # bias
            None,               # handle
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
