import importlib

import torch


_FF_EXT = False
_NEXT_HANDLE = 1
_REGISTERED_HANDLES: set[int] = set()
_BIT_GRAD_CACHE: dict[int, torch.Tensor] = {}
_EXT_CANDIDATES = ("firefly_ext", "firefly_bitnet_ext")


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


def _to_ternary_codes(weight: torch.Tensor, threshold: float = 0.0) -> torch.Tensor:
    thr = float(threshold)
    pos = weight > thr
    neg = weight < -thr
    codes = torch.zeros_like(weight, dtype=torch.uint8)
    codes[pos] = 1
    codes[neg] = 2
    return codes


def pack_ternary_weight(weight: torch.Tensor, threshold: float = 0.0) -> torch.Tensor:
    if weight.ndim != 2:
        raise ValueError(
            f"weight must be 2D [out_features, in_features], got {tuple(weight.shape)}"
        )
    out_features, in_features = weight.shape
    codes = _to_ternary_codes(weight, threshold=threshold)
    pad = (-in_features) % 4
    if pad:
        codes = torch.nn.functional.pad(codes, (0, pad), value=0)
    codes = codes.view(out_features, -1, 4).to(torch.int16)
    packed = (
        codes[..., 0]
        | (codes[..., 1] << 2)
        | (codes[..., 2] << 4)
        | (codes[..., 3] << 6)
    ).to(torch.uint8)
    return packed.contiguous()


def _get_firefly_bitnet_ext():
    global _FF_EXT
    if _FF_EXT is not False:
        return _FF_EXT
    for mod_name in _EXT_CANDIDATES:
        try:
            _FF_EXT = importlib.import_module(mod_name)
            break
        except Exception:
            _FF_EXT = None
    return _FF_EXT


def _packed_linear_forward_fallback(
    x_f32: torch.Tensor, packed_weight: torch.Tensor, in_features: int, scale: float
) -> torch.Tensor:
    out_features, packed_cols = packed_weight.shape
    expected_cols = (in_features + 3) // 4
    if packed_cols != expected_cols:
        raise ValueError(
            f"packed_weight second dim mismatch: expected {expected_cols}, got {packed_cols}"
        )

    out = x_f32.new_zeros((x_f32.shape[0], out_features))
    code_to_val = torch.tensor(
        [0.0, 1.0, -1.0, 0.0], device=x_f32.device, dtype=x_f32.dtype
    )
    pw = packed_weight.to(device=x_f32.device)

    for shift in range(4):
        cols = x_f32[:, shift:in_features:4]
        if cols.shape[1] == 0:
            continue
        codes = ((pw >> (2 * shift)) & 0x3).to(torch.long)
        w_shift = code_to_val[codes[:, : cols.shape[1]]]
        out.addmm_(cols, w_shift.t())

    if scale != 1.0:
        out.mul_(scale)
    return out


def _packed_linear_backward_input_fallback(
    grad_out_f32: torch.Tensor,
    packed_weight: torch.Tensor,
    in_features: int,
    scale: float,
) -> torch.Tensor:
    batch = grad_out_f32.shape[0]
    grad_x = grad_out_f32.new_zeros((batch, in_features))
    code_to_val = torch.tensor(
        [0.0, 1.0, -1.0, 0.0], device=grad_out_f32.device, dtype=grad_out_f32.dtype
    )
    pw = packed_weight.to(device=grad_out_f32.device)

    for shift in range(4):
        k_idx = torch.arange(shift, in_features, 4, device=grad_out_f32.device)
        if k_idx.numel() == 0:
            continue
        codes = ((pw[:, : k_idx.numel()] >> (2 * shift)) & 0x3).to(torch.long)
        w_shift = code_to_val[codes]
        grad_x[:, k_idx] = grad_out_f32.matmul(w_shift)

    if scale != 1.0:
        grad_x.mul_(scale)
    return grad_x


class PackedBitLinearFn(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x2d: torch.Tensor,
        packed_weight: torch.Tensor,
        in_features: int,
        scale: float,
        handle: int,
    ):
        ext = _get_firefly_bitnet_ext()
        x_f32 = x2d.contiguous().float()
        if ext is None:
            out = _packed_linear_forward_fallback(
                x_f32, packed_weight.contiguous(), int(in_features), float(scale)
            )
        else:
            out = ext.ff_packed_linear_forward(
                x_f32, packed_weight.contiguous(), int(in_features), float(scale)
            )
        ctx.save_for_backward(x_f32, packed_weight)
        ctx.in_features = int(in_features)
        ctx.scale = float(scale)
        ctx.handle = int(handle)
        ctx.input_dtype = x2d.dtype
        return out

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        ext = _get_firefly_bitnet_ext()
        x_f32, packed_weight = ctx.saved_tensors
        grad_out_f32 = grad_out.contiguous().float()

        if ext is None:
            grad_in = _packed_linear_backward_input_fallback(
                grad_out_f32,
                packed_weight.contiguous(),
                int(ctx.in_features),
                float(ctx.scale),
            )
        else:
            grad_in = ext.ff_packed_linear_backward_input(
                grad_out_f32,
                packed_weight.contiguous(),
                int(ctx.in_features),
                float(ctx.scale),
            )
        grad_w = grad_out_f32.t().matmul(x_f32) * float(ctx.scale)
        _BIT_GRAD_CACHE[int(ctx.handle)] = grad_w

        return grad_in.to(dtype=ctx.input_dtype), None, None, None, None


@torch.no_grad()
def update_packed_ternary_weight_(
    packed_weight: torch.Tensor,
    direction: torch.Tensor,
    mask: torch.Tensor,
    in_features: int,
) -> None:
    if direction.dtype not in (torch.float32, torch.float16, torch.bfloat16):
        direction = direction.float()
    if mask.dtype != torch.bool:
        mask = mask.bool()

    out_features = packed_weight.shape[0]
    if direction.shape != (out_features, in_features):
        raise ValueError(
            f"direction shape mismatch: expected {(out_features, in_features)}, got {tuple(direction.shape)}"
        )
    if mask.shape != (out_features, in_features):
        raise ValueError(
            f"mask shape mismatch: expected {(out_features, in_features)}, got {tuple(mask.shape)}"
        )

    for shift in range(4):
        lane_cols = (in_features - shift + 3) // 4
        if lane_cols <= 0:
            continue
        src = packed_weight[:, :lane_cols]
        codes = ((src >> (2 * shift)) & 0x3).to(torch.uint8)

        d_lane = direction[:, shift:in_features:4]
        m_lane = mask[:, shift:in_features:4].bool()
        active = m_lane & (d_lane != 0)

        pos_dir = active & (d_lane > 0)
        neg_dir = active & (d_lane < 0)
        new_codes = codes.clone()

        pos_drop = pos_dir & (codes == 1)  # +1 -> 0
        pos_to_neg = pos_dir & (codes == 0)  # 0 -> -1
        neg_raise = neg_dir & (codes == 2)  # -1 -> 0
        neg_to_pos = neg_dir & (codes == 0)  # 0 -> +1

        new_codes[pos_drop] = 0
        new_codes[pos_to_neg] = 2
        new_codes[neg_raise] = 0
        new_codes[neg_to_pos] = 1

        clear_mask = torch.tensor(
            0xFF ^ (0x3 << (2 * shift)), device=packed_weight.device, dtype=torch.uint8
        )
        merged = (src & clear_mask) | (new_codes << (2 * shift))
        packed_weight[:, :lane_cols].copy_(merged)
