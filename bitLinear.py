import math

import torch
from torch import nn

from .fireflykernels import (
    PackedBitLinearFn,
    consume_bit_grad,
    next_bit_handle,
    pack_ternary_weight,
    register_bit_handle,
    release_bit_handle,
)


def _init_packed_ternary_weight(
    out_features: int, in_features: int, prob: float, device: torch.device
) -> torch.Tensor:
    if not (0.0 <= prob <= 1.0):
        raise ValueError(f"prob must be in [0, 1], got {prob}")
    padded_in = ((in_features + 3) // 4) * 4
    r = torch.rand((out_features, in_features), device=device)
    half = prob * 0.5
    codes = torch.zeros((out_features, padded_in), dtype=torch.uint8, device=device)
    codes[:, :in_features][r < half] = 1
    codes[:, :in_features][(r >= half) & (r < prob)] = 2
    codes = codes.view(out_features, -1, 4).to(torch.int16)
    packed = (
        codes[..., 0]
        | (codes[..., 1] << 2)
        | (codes[..., 2] << 4)
        | (codes[..., 3] << 6)
    ).to(torch.uint8)
    return packed.contiguous()


class BitLinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        threshold: float = 0.0,
    ) -> None:
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.threshold = float(threshold)
        self.scale = 1.0 / math.sqrt(in_features)

        if self.threshold == 0.0:
            packed_init = _init_packed_ternary_weight(
                out_features=self.out_features,
                in_features=self.in_features,
                prob=0.3,
                device=torch.device("cpu"),
            )
        else:
            init_w = torch.zeros((out_features, in_features), dtype=torch.float32)
            packed_init = pack_ternary_weight(init_w, threshold=self.threshold)
        self.register_buffer(
            "packed_weight",
            packed_init,
            persistent=True,
        )
        self.register_buffer(
            "_bit_handle",
            torch.tensor(next_bit_handle(), dtype=torch.int64),
            persistent=True,
        )
        self._registered_handle = int(self._bit_handle.item())
        self.register_load_state_dict_post_hook(self._post_load_state_dict)
        self.channel_scale = nn.Parameter(torch.ones(out_features))

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)

    @torch.no_grad()
    def reset_ternary_(self, prob: float = 0.3) -> None:
        self.packed_weight.copy_(
            _init_packed_ternary_weight(
                out_features=self.out_features,
                in_features=self.in_features,
                prob=prob,
                device=self.packed_weight.device,
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != self.in_features:
            raise ValueError(
                f"expected input last dim {self.in_features}, got {x.shape[-1]}"
            )
        if self.packed_weight.device != x.device:
            raise RuntimeError(
                "BitLinear input and packed_weight must be on same device. "
                "Move module with model.to(device) before forward."
            )
        x2d = x.reshape(-1, self.in_features)
        out2d = PackedBitLinearFn.apply(
            x2d,
            self.packed_weight,
            self.in_features,
            self.scale,
            int(self._bit_handle.item()),
        )
        if self.channel_scale.device != out2d.device:
            raise RuntimeError(
                "BitLinear channel_scale and output must be on same device. "
                "Move module with model.to(device) before forward."
            )
        out2d = out2d * self.channel_scale.to(dtype=out2d.dtype)
        if self.bias is not None:
            if self.bias.device != out2d.device:
                raise RuntimeError(
                    "BitLinear bias and output must be on same device. "
                    "Move module with model.to(device) before forward."
                )
            out2d = out2d + self.bias.to(dtype=out2d.dtype)
        return out2d.view(*x.shape[:-1], self.out_features)

    def consume_weight_grad(self) -> torch.Tensor | None:
        return consume_bit_grad(int(self._bit_handle.item()))

    def _post_load_state_dict(self, module, incompatible_keys) -> None:
        del module, incompatible_keys
        new_handle = register_bit_handle(int(self._bit_handle.item()))
        if new_handle != self._registered_handle:
            release_bit_handle(self._registered_handle)
            self._registered_handle = new_handle


def collect_bitlinear_modules(module: nn.Module) -> list[BitLinear]:
    return [m for m in module.modules() if isinstance(m, BitLinear)]
