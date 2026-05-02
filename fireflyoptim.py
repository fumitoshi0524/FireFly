import torch
from bitsandbytes.optim import AdamW8bit

from .bitLinear import BitLinear
from .fireflykernels import update_int8_weight_, _invalidate_weight_cache


class FireFlyOptim(AdamW8bit):
    """bnb AdamW8bit + DQT stochastic rounding for INT8 weights.

    Dense parameters (RMSNorm, lm_head, biases) use standard bnb 8-bit AdamW.
    INT8 weights use the SAME lr / betas / weight_decay, then stochastic rounding
    snaps them back to int8 (matching the DQT paper).
    """

    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=1e-1,
        amsgrad=False,
        optim_bits=8,
        min_8bit_size=4096,
        percentile_clipping=100,
        block_wise=True,
        is_paged=False,
        theta=0.0,
        bit_modules=None,
    ):
        super().__init__(
            params,
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            amsgrad=amsgrad,
            optim_bits=optim_bits,
            args=None,
            min_8bit_size=min_8bit_size,
            percentile_clipping=percentile_clipping,
            block_wise=block_wise,
            is_paged=is_paged,
        )
        self.bit_modules = list(bit_modules) if bit_modules is not None else []
        self._bit_handles = (
            [int(m._bit_handle.item()) for m in self.bit_modules]
            if bit_modules is not None
            else []
        )
        self._bit_state: dict[int, dict[str, torch.Tensor | int]] = {}
        self.theta = float(theta)

    def add_bit_modules(self, modules) -> None:
        for module in modules:
            if not isinstance(module, BitLinear):
                raise TypeError(f"expected BitLinear, got {type(module).__name__}")
            if module not in self.bit_modules:
                self.bit_modules.append(module)
                self._bit_handles.append(int(module._bit_handle.item()))

    @torch.no_grad()
    def step(self, closure=None):
        # 1. Standard AdamW for dense params (RMSNorm, lm_head, biases)
        loss = super().step(closure)

        if not self.bit_modules:
            return loss

        # 2. DQT stochastic rounding for INT8 weights
        # Uses the SAME lr, betas, weight_decay as the dense AdamW
        group = self.param_groups[0]
        lr = float(group["lr"])
        beta1, beta2 = group["betas"]
        eps = float(group["eps"])
        wd = float(group["weight_decay"])

        for module, handle in zip(self.bit_modules, self._bit_handles):
            g = module.consume_weight_grad()
            if g is None:
                continue
            g = g.float().contiguous()

            state = self._bit_state.setdefault(handle, {})
            if "m" not in state or state["m"].shape != g.shape:
                state["m"] = torch.zeros_like(g, dtype=torch.float32)
                state["v"] = torch.zeros_like(g, dtype=torch.float32)
                state["t"] = 0
            if "residual" not in state or state["residual"].shape != g.shape:
                state["residual"] = torch.zeros_like(g, dtype=torch.float32)

            m, v, residual = state["m"], state["v"], state["residual"]

            state["t"] += 1
            # AdamW update: m = b1*m + (1-b1)*g, v = b2*v + (1-b2)*g^2
            m.mul_(beta1).add_(g, alpha=1.0 - beta1)
            v.mul_(beta2).addcmul_(g, g, value=1.0 - beta2)
            m_hat = m / (1.0 - beta1 ** state["t"])
            v_hat = v / (1.0 - beta2 ** state["t"])

            # Dense AdamW step: ΔW_eff = -lr * (m_hat / (sqrt(v_hat) + eps) + wd * W_eff)
            # Convert to INT8 units: residual += ΔW_eff / weight_scale
            ws = module.weight_scale.float().unsqueeze(1).clamp_min(eps)
            iw = module.int_weight.float()
            w_eff = iw * ws  # current effective weight
            adam_term = m_hat / (v_hat.sqrt() + eps)
            delta_w_eff = -lr * (adam_term + wd * w_eff)
            residual.add_(delta_w_eff / ws)

            # Stochastic rounding: when |residual| >= 1, flip int_weight
            abs_res = residual.abs()
            base = torch.floor(abs_res)
            frac = abs_res - base
            extra = (torch.rand_like(frac) < frac).float()
            delta_q = (torch.sign(residual) * (base + extra)).to(torch.int32)

            if torch.any(delta_q != 0):
                update_int8_weight_(module.int_weight, delta_q)
                residual.sub_(delta_q.float())
                _invalidate_weight_cache(handle)

        return loss

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
        bit_state = state_dict.pop("bit_state", {})
        super().load_state_dict(state_dict)
        self._bit_state = {}
        for handle, per_handle in bit_state.items():
            self._bit_state[int(handle)] = {
                key: value.clone() if torch.is_tensor(value) else value
                for key, value in per_handle.items()
            }
