import torch
from torch.optim import Optimizer

from .fireflykernels import update_packed_ternary_weight_
from .bitLinear import BitLinear


class FireFlyProb(Optimizer):
    def __init__(
        self,
        params,
        base_ratio=0.002,
        lr_dense=1e-3,
        vote_interval=4,
        vote_threshold=3,
        vote_scale=1.0,
        clip_grad=1.0,
        bit_modules=None,
    ):
        defaults = dict(
            base_ratio=base_ratio,
            lr_dense=lr_dense,
            vote_interval=vote_interval,
            vote_threshold=vote_threshold,
            vote_scale=vote_scale,
            clip_grad=clip_grad,
        )
        super().__init__(params, defaults)
        self.bit_modules = list(bit_modules) if bit_modules is not None else []
        self._bit_state: dict[int, dict[str, torch.Tensor]] = {}
        self._bit_step = 0

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
            base_ratio = cfg["base_ratio"]
            vote_interval = max(1, int(cfg["vote_interval"]))
            vote_threshold = min(vote_interval, max(1, int(cfg["vote_threshold"])))
            vote_scale = max(1.0, float(cfg.get("vote_scale", 1.0)))
            self._bit_step += max(1, int(round(vote_scale)))
            do_vote_update = (self._bit_step % vote_interval) == 0

            for module in self.bit_modules:
                g = module.consume_weight_grad()
                if g is None:
                    continue
                g = g.float()

                handle = int(module._bit_handle.item())
                state = self._bit_state.setdefault(handle, {})
                if "m" not in state or state["m"].shape != g.shape:
                    state["m"] = torch.zeros_like(g, dtype=torch.float16)
                elif state["m"].device != g.device:
                    state["m"] = state["m"].to(device=g.device, dtype=torch.float16)
                m = state["m"]

                g_abs = g.abs()
                mag_norm = (g_abs / (g_abs.mean() + 1e-8)).clamp_(0.0, 4.0)
                signed_vote = (torch.sign(g) * mag_norm * vote_scale).to(m.dtype)
                m.add_(signed_vote)
                vote_clip = float(vote_interval)
                m.clamp_(-vote_clip, vote_clip)

                if not do_vote_update:
                    continue

                direction = torch.sign(m.float())
                confidence = m.abs().float()
                eligible = confidence >= float(vote_threshold)
                if not torch.any(eligible):
                    continue

                prob = confidence / (confidence.mean() + 1e-8)
                prob = (prob * mag_norm * base_ratio).clamp_(0, 1)
                mask = (torch.rand_like(prob) < prob) & eligible
                if not torch.any(mask):
                    continue

                update_packed_ternary_weight_(
                    packed_weight=module.packed_weight,
                    direction=direction,
                    mask=mask,
                    in_features=module.in_features,
                )

                m_update = (m[mask].float() - direction[mask] * float(vote_threshold)).clamp_(
                    -vote_clip, vote_clip
                )
                m[mask] = m_update.to(m.dtype)

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is not None:
                    p.grad.zero_()

    def state_dict(self):
        state = super().state_dict()
        state["bit_step"] = int(self._bit_step)
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
        bit_step = int(state_dict.pop("bit_step", 0))
        bit_state = state_dict.pop("bit_state", {})
        super().load_state_dict(state_dict)
        self._bit_step = bit_step
        self._bit_state = {
            int(handle): {
                key: value.clone() if torch.is_tensor(value) else value
                for key, value in per_handle.items()
            }
            for handle, per_handle in bit_state.items()
        }
