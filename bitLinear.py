import math

import torch
from torch import nn

from .fireflykernels import (
    Int8LinearFn,
    consume_bit_grad,
    next_bit_handle,
    quantize_fp_to_int8,
    register_bit_handle,
    release_bit_handle,
)


def _init_int8_weight(out_features: int, in_features: int):
    weight = torch.empty((out_features, in_features), dtype=torch.float32)
    nn.init.kaiming_uniform_(weight, a=math.sqrt(5))
    q, scale = quantize_fp_to_int8(weight)
    scale = scale * math.sqrt(in_features)
    return q, scale


class BitLinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        threshold: float = 0.0,
        n0prob: float = 0.3,
    ) -> None:
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.threshold = float(threshold)
        self.scale = 1.0 / math.sqrt(in_features)
        self.n0prob = float(n0prob)

        int_init, scale_init = _init_int8_weight(
            out_features=self.out_features, in_features=self.in_features
        )
        self.register_buffer(
            "int_weight",
            int_init,
            persistent=True,
        )
        self.weight_scale = nn.Parameter(scale_init)
        self.register_buffer(
            "_bit_handle",
            torch.tensor(next_bit_handle(), dtype=torch.int64),
            persistent=True,
        )
        self._registered_handle = int(self._bit_handle.item())
        self.register_load_state_dict_post_hook(self._post_load_state_dict)

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)

    @torch.no_grad()
    def reset_int8_(self) -> None:
        int_init, scale_init = _init_int8_weight(
            out_features=self.out_features, in_features=self.in_features
        )
        self.int_weight.copy_(int_init.to(device=self.int_weight.device))
        self.weight_scale.copy_(scale_init.to(device=self.weight_scale.device))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != self.in_features:
            raise ValueError(
                f"expected input last dim {self.in_features}, got {x.shape[-1]}"
            )
        if self.int_weight.device != x.device:
            raise RuntimeError(
                "BitLinear input and int_weight must be on same device. "
                "Move module with model.to(device) before forward."
            )
        x2d = x.reshape(-1, self.in_features)
        out2d = Int8LinearFn.apply(
            x2d,
            self.int_weight,
            self.weight_scale,
            self.bias,
            int(self._bit_handle.item()),
        )
        out2d = out2d * self.scale
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
