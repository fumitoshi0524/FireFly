import torch


# ---------------------------------------------------------------------------
# Handle registry
# ---------------------------------------------------------------------------
_NEXT_HANDLE = 1
_REGISTERED_HANDLES: set[int] = set()
_BIT_GRAD_CACHE: dict[int, torch.Tensor] = {}


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


# ---------------------------------------------------------------------------
# bitsandbytes backend
# ---------------------------------------------------------------------------
_BNB = None
_BNB_F = None
_BNB_FMT = "col_ampere"

try:
    import bitsandbytes as _BNB
    import bitsandbytes.functional as _BNB_F
    if torch.cuda.is_available():
        cc = torch.cuda.get_device_capability()
        if cc[0] >= 8:
            _BNB_FMT = "col_ampere"
        elif cc[0] == 7 and cc[1] >= 5:
            _BNB_FMT = "col_turing"
        else:
            _BNB_FMT = "col32"
except ImportError:
    pass

# Weight-transform cache  (handle → (CxB, SB, version))
_BNB_WCACHE: dict[int, tuple] = {}
_BNB_WVERSION: dict[int, int] = {}


def _cached_weight_transform(
    handle: int, int_weight: torch.Tensor, weight_scale: torch.Tensor
):
    """Return (CxB, SB) for the current weight; re-transform only if stale."""
    cur_ver = _BNB_WVERSION.get(handle, 0)
    entry = _BNB_WCACHE.get(handle)
    if entry is not None:
        CxB, SB, cached_ver = entry
        if cached_ver == cur_ver:
            return CxB, SB

    # bitsandbytes expects fp16 input for int8_vectorwise_quant
    w_fp16 = int_weight.to(torch.float16) * weight_scale.to(torch.float16).unsqueeze(1)
    w_q, _w_s = _BNB_F.int8_vectorwise_quant(w_fp16)
    CxB, SB = _BNB_F.transform(w_q, _BNB_FMT)
    _BNB_WCACHE[handle] = (CxB, SB, cur_ver)
    return CxB, SB


def _invalidate_weight_cache(handle: int):
    """Call after int_weight is modified by stochastic rounding."""
    _BNB_WVERSION[handle] = _BNB_WVERSION.get(handle, 0) + 1


# ---------------------------------------------------------------------------
# Weight quantisation  (static, used at init / reset)
# ---------------------------------------------------------------------------
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
# Int8LinearFn
# ---------------------------------------------------------------------------
class Int8LinearFn(torch.autograd.Function):

    @staticmethod
    def forward(
        ctx,
        x2d: torch.Tensor,          # [N, K]  (autocast doesn't cover custom Fns)
        int_weight: torch.Tensor,   # [O, K]  int8
        weight_scale: torch.Tensor, # [O]     float  (trainable Parameter)
        bias: torch.Tensor | None,  # [O]     float
        handle: int,
    ):
        if _BNB_F is not None:
            return _forward_bnb(ctx, x2d, int_weight, weight_scale, bias, handle)
        return _forward_bf16(ctx, x2d, int_weight, weight_scale, bias, handle)

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        if _BNB_F is not None:
            return _backward_bnb(ctx, grad_out)
        return _backward_bf16(ctx, grad_out)


# ---------------------------------------------------------------------------
# Path A: bitsandbytes cuBLASLt INT8 Tensor Cores
# ---------------------------------------------------------------------------

def _forward_bnb(ctx, x2d, int_weight, weight_scale, bias, handle):
    # 1. Quantise activation — bitsandbytes works in fp16
    x_fp16 = x2d.half()
    CA, SCA = _BNB_F.int8_vectorwise_quant(x_fp16)

    # 2. Transform activation to col32 layout
    C32A, SA = _BNB_F.transform(CA, "col32")

    # 3. Get cached weight transform
    CxB, SB = _cached_weight_transform(handle, int_weight, weight_scale)

    # 4. INT8 matmul via cuBLASLt
    out_i32, _ = _BNB_F.igemmlt(C32A, CxB, SA, SB)

    # 5. Dequantise — output must match activation scale × weight scale
    out = _BNB_F.mm_dequant(out_i32, SCA, weight_scale.half().unsqueeze(1))

    if bias is not None:
        out.add_(bias.half())

    # Save quantised activation for backward
    ctx.save_for_backward(CA, SCA, int_weight, weight_scale)
    ctx.handle = int(handle)
    ctx.has_bias = bias is not None
    ctx.input_dtype = x2d.dtype
    return out.to(dtype=x2d.dtype)


def _backward_bnb(ctx, grad_out):
    CA, SCA, int_weight, weight_scale = ctx.saved_tensors
    go_fp16 = grad_out.half()
    ws_fp16 = weight_scale.half()
    handle = ctx.handle

    CxB, SB = _cached_weight_transform(handle, int_weight, weight_scale)

    # --- grad_in = grad_out @ W  (use igemmlt with same weight layout) ---
    Cgo, SCgo = _BNB_F.int8_vectorwise_quant(go_fp16)
    C32go, Sgo = _BNB_F.transform(Cgo, "col32")
    grad_in_i32, _ = _BNB_F.igemmlt(C32go, CxB, Sgo, SB)
    grad_in = _BNB_F.mm_dequant(grad_in_i32, SCgo, ws_fp16.unsqueeze(0))

    # --- grad_w = grad_out^T @ x  (swap operands: x_q as A, go as B) ---
    # Save x_q (int8) from forward → transform, then igemmlt(go_col32, xq_col32, transposed_B=True)
    C32x, Sx = _BNB_F.transform(CA, "col32")  # CA is the saved int8 activation

    Cg2, SCg2 = _BNB_F.int8_vectorwise_quant(go_fp16)
    C32g2, Sg2 = _BNB_F.transform(Cg2, "col32")

    grad_w_i32, _ = _BNB_F.igemmlt(C32g2, C32x, Sg2, Sx)
    grad_w = _BNB_F.mm_dequant(grad_w_i32, SCg2, SCA)

    cached = _BIT_GRAD_CACHE.get(handle)
    if cached is None:
        _BIT_GRAD_CACHE[handle] = grad_w.to(dtype=torch.bfloat16)
    else:
        cached.add_(grad_w.to(dtype=torch.bfloat16))

    # --- grad_weight_scale  =  (grad_w ⊙ int_weight).sum(dim=1) ---
    grad_weight_scale = (
        grad_w.float() * int_weight.float()
    ).sum(dim=1).to(dtype=weight_scale.dtype)

    grad_bias = go_fp16.sum(dim=0).to(dtype=grad_out.dtype) if ctx.has_bias else None

    return (
        grad_in.to(dtype=grad_out.dtype),
        None,
        grad_weight_scale,
        grad_bias,
        None,
    )


# ---------------------------------------------------------------------------
# Path B: BF16 fallback  (torch.matmul on BF16 Tensor Cores, works everywhere)
# ---------------------------------------------------------------------------

def _forward_bf16(ctx, x2d, int_weight, weight_scale, bias, handle):
    x2d_bf16 = x2d.to(torch.bfloat16)
    w_bf16 = int_weight.to(torch.bfloat16) * weight_scale.to(torch.bfloat16).unsqueeze(1)
    out = torch.matmul(x2d_bf16, w_bf16.t())
    if bias is not None:
        out.add_(bias)
    ctx.save_for_backward(x2d_bf16, int_weight, weight_scale)
    ctx.handle = int(handle)
    ctx.has_bias = bias is not None
    ctx.input_dtype = x2d.dtype
    return out


def _backward_bf16(ctx, grad_out):
    x2d_bf16, int_weight, weight_scale = ctx.saved_tensors
    go_bf16 = grad_out.to(torch.bfloat16)
    ws_bf16 = weight_scale.to(torch.bfloat16)

    w_bf16 = int_weight.to(torch.bfloat16) * ws_bf16.unsqueeze(1)

    grad_in = torch.matmul(go_bf16, w_bf16)

    grad_w = torch.matmul(go_bf16.t(), x2d_bf16)
    cached = _BIT_GRAD_CACHE.get(ctx.handle)
    if cached is None:
        _BIT_GRAD_CACHE[ctx.handle] = grad_w.to(dtype=torch.bfloat16)
    else:
        cached.add_(grad_w.to(dtype=torch.bfloat16))

    grad_weight_scale = (
        grad_w.float() * int_weight.float()
    ).sum(dim=1).to(dtype=weight_scale.dtype)

    grad_bias = go_bf16.sum(dim=0).to(dtype=grad_out.dtype) if ctx.has_bias else None

    return (
        grad_in.to(dtype=grad_out.dtype),
        None,
        grad_weight_scale,
        grad_bias,
        None,
    )


# ---------------------------------------------------------------------------
# INT8 weight update
# ---------------------------------------------------------------------------

@torch.no_grad()
def update_int8_weight_(int_weight: torch.Tensor, delta_q: torch.Tensor) -> None:
    """In-place int8 weight update:  W += delta_q, clamped to [-127, 127]."""
    result = int_weight.to(torch.int16) + delta_q.to(torch.int16)
    result.clamp_(-127, 127)
    int_weight.copy_(result.to(torch.int8))
