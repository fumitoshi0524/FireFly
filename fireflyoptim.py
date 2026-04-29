import torch
from torch.optim import Optimizer

from .fireflykernels import update_packed_ternary_weight_
from .bitLinear import BitLinear


class FireFlyOptim(Optimizer):
    def __init__(
        self,
        params,
        lr_ternary=0.01,
        lr_dense=1e-3,
        clip_grad=1.0,
        theta=0.1,
        bit_modules=None,
    ):
        defaults = dict(
            lr_ternary=lr_ternary,
            lr_dense=lr_dense,
            clip_grad=clip_grad,
            theta=theta,
        )
        super().__init__(params, defaults)
        self.bit_modules = list(bit_modules) if bit_modules is not None else []
        self._bit_state: dict[int, dict[str, torch.Tensor]] = {}

    def add_bit_modules(self, modules) -> None:
        for module in modules:
            if not isinstance(module, BitLinear):
                raise TypeError(f"expected BitLinear, got {type(module).__name__}")
            if module not in self.bit_modules:
                self.bit_modules.append(module)

    @torch.no_grad()
    def step(self):
        dense_params = [
            p for g in self.param_groups for p in g["params"] if p.grad is not None
        ]
        if dense_params:
            torch.nn.utils.clip_grad_norm_(
                dense_params, max_norm=self.defaults["clip_grad"]
            )
        for group in self.param_groups:
            lr_dense = group.get("lr", group["lr_dense"])

            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                state = self.state[p]
                if "m" not in state:
                    state["m"] = torch.zeros_like(p)
                    state["v"] = torch.zeros_like(p)
                    state["t"] = 0
                m, v = state["m"], state["v"]
                state["t"] += 1
                beta1, beta2 = 0.9, 0.95
                m.mul_(beta1).add_(g, alpha=1 - beta1)
                v.mul_(beta2).addcmul_(g, g, value=1 - beta2)
                m_hat = m / (1 - beta1 ** state["t"])
                v_hat = v / (1 - beta2 ** state["t"])
                p.add_(m_hat / (v_hat.sqrt() + 1e-8), alpha=-lr_dense)

        if self.bit_modules:
            cfg = self.param_groups[0]
            lr_ternary = cfg["lr_ternary"]
            theta = cfg["theta"]

            for module in self.bit_modules:
                g = module.consume_weight_grad()
                if g is None:
                    continue
                g = g.float()

                handle = int(module._bit_handle.item())
                state = self._bit_state.setdefault(handle, {})
                if "m" not in state or state["m"].shape != g.shape:
                    state["m"] = torch.zeros_like(g, dtype=torch.bfloat16)
                    state["v"] = torch.zeros_like(g, dtype=torch.bfloat16)
                    state["residual"] = torch.zeros_like(g, dtype=torch.bfloat16)
                    state["t"] = 0
                elif (
                    state["m"].device != g.device
                    or state["m"].dtype != torch.bfloat16
                    or state["v"].dtype != torch.bfloat16
                ):
                    state["m"] = state["m"].to(device=g.device, dtype=torch.bfloat16)
                    state["v"] = state["v"].to(device=g.device, dtype=torch.bfloat16)
                    state["residual"] = state["residual"].to(device=g.device, dtype=torch.bfloat16)

                m, v, residual = state["m"], state["v"], state["residual"]
                state["t"] += 1
                g_bf16 = g.to(dtype=torch.bfloat16)
                m.mul_(0.9).add_(g_bf16, alpha=0.1)
                v.mul_(0.95).addcmul_(g_bf16, g_bf16, value=0.05)

                m_hat = m.float() / (1 - 0.9 ** state["t"])
                v_hat = v.float() / (1 - 0.95 ** state["t"])

                # OU residual: accumulate Adam update, apply mean reversion
                # residual tracks the continuous offset virtual_w - w_ternary
                adam_update = lr_ternary * m_hat / (v_hat.sqrt() + 1e-8)
                residual.add_(adam_update.to(dtype=torch.bfloat16))
                residual.mul_(1.0 - theta)

                direction = torch.sign(residual.float())
                prob = residual.float().abs().clamp(0.0, 1.0)
                mask = (torch.rand_like(prob) < prob) & (direction != 0)
                if not torch.any(mask):
                    continue

                update_packed_ternary_weight_(
                    packed_weight=module.packed_weight,
                    direction=direction,
                    mask=mask,
                    in_features=module.in_features,
                )

                # residual_new = residual_old - Δw
                # kernel flips: Δw = -direction → residual += direction
                residual[mask] += direction[mask].to(dtype=torch.bfloat16)

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is not None:
                    p.grad.zero_()

    def state_dict(self):
        state = super().state_dict()
        state["bit_state"] = {
            int(handle): {
                key: value.clone() if torch.is_tensor(value) else value
                for key, value in per_handle.items()
            }
            for handle, per_handle in self._bit_state.items()
        }
        return state

    def load_state_dict(self, state_dict):
        state_dict = dict(state_dict)
        state_dict.pop("bit_step", None)
        bit_state = state_dict.pop("bit_state", {})
        super().load_state_dict(state_dict)
        self._bit_state = {}
        for handle, per_handle in bit_state.items():
            per_handle.pop("score_ema", None)  # backward-compat
            self._bit_state[int(handle)] = {
                key: value.clone() if torch.is_tensor(value) else value
                for key, value in per_handle.items()
            }
