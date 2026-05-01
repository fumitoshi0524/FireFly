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


class Int8LinearFn(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x2d: torch.Tensor,
        int_weight: torch.Tensor,
        weight_scale: torch.Tensor,
        channel_scale: torch.Tensor,
        bias: torch.Tensor | None,
        handle: int,
    ):
        ext = _get_firefly_ext()
        x_saved = x2d.contiguous().float()
        x_q, x_scale = _quantize_activation_per_token(x_saved)
        if ext is not None and hasattr(ext, "ff_int8_linear_forward"):
            out_i32 = ext.ff_int8_linear_forward(x_q.contiguous(), int_weight.contiguous())
        else:
            out_i32 = x_q.to(torch.int32).matmul(int_weight.to(torch.int32).t())
        out = out_i32.float() * (x_scale * weight_scale.float().view(1, -1))
        out.mul_(channel_scale.float().view(1, -1))
        if bias is not None:
            out.add_(bias.float().view(1, -1))

        ctx.save_for_backward(x_saved, int_weight, weight_scale, channel_scale)
        ctx.handle = int(handle)
        ctx.input_dtype = x2d.dtype
        ctx.has_bias = bias is not None
        return out.to(dtype=x2d.dtype)

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        ext = _get_firefly_ext()
        x_saved, int_weight, weight_scale, channel_scale = ctx.saved_tensors
        grad_out_f32 = grad_out.contiguous().float()

        if ext is not None and hasattr(ext, "ff_int8_linear_backward_input"):
            grad_in = ext.ff_int8_linear_backward_input(
                grad_out_f32,
                int_weight.contiguous(),
                weight_scale.float().contiguous(),
                channel_scale.float().contiguous(),
            )
        else:
            w_deq = int_weight.float() * weight_scale.float().unsqueeze(1)
            w_eff = w_deq * channel_scale.float().unsqueeze(1)
            grad_in = grad_out_f32.matmul(w_eff)

        if ext is not None and hasattr(ext, "ff_int8_linear_backward_weight"):
            grad_w = ext.ff_int8_linear_backward_weight(grad_out_f32, x_saved)
        else:
            grad_w = grad_out_f32.t().matmul(x_saved)
        grad_w.mul_(channel_scale.float().unsqueeze(1))
        cached = _BIT_GRAD_CACHE.get(ctx.handle)
        if cached is None:
            _BIT_GRAD_CACHE[ctx.handle] = grad_w.to(dtype=torch.bfloat16)
        else:
            cached.add_(grad_w.to(dtype=torch.bfloat16))

        w_deq = int_weight.float() * weight_scale.float().unsqueeze(1)
        pre_channel = x_saved.matmul(w_deq.t())
        grad_channel_scale = (grad_out_f32 * pre_channel).sum(dim=0).to(
            dtype=channel_scale.dtype
        )
        grad_bias = (
            grad_out_f32.sum(dim=0).to(dtype=grad_out.dtype) if ctx.has_bias else None
        )
        return (
            grad_in.to(dtype=ctx.input_dtype),
            None,
            None,
            grad_channel_scale,
            grad_bias,
            None,
        )


@torch.no_grad()
def update_int8_weight_(int_weight: torch.Tensor, delta_q: torch.Tensor) -> None:
    if int_weight.dtype != torch.int8:
        raise TypeError(f"int_weight must be torch.int8, got {int_weight.dtype}")
    if delta_q.shape != int_weight.shape:
        raise ValueError(
            f"delta_q shape mismatch: expected {tuple(int_weight.shape)}, got {tuple(delta_q.shape)}"
        )
    next_weight = int_weight.to(torch.int16).add_(delta_q.to(torch.int16))
    next_weight.clamp_(-127, 127)
    int_weight.copy_(next_weight.to(torch.int8))
