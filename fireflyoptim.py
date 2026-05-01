import torch
from torch.optim import Optimizer

from .bitLinear import BitLinear
from .fireflykernels import update_int8_weight_, _invalidate_weight_cache


def _pad_to_block_multiple(t: torch.Tensor, blocksize: int, num_blocks: int):
    """Pad 1-D tensor to num_blocks * blocksize with zeros."""
    padded_size = num_blocks * blocksize
    if padded_size > t.numel():
        return torch.cat(
            [
                t,
                torch.zeros(padded_size - t.numel(), device=t.device, dtype=t.dtype),
            ]
        )
    return t


def _quantize_blockwise_signed(x: torch.Tensor, blocksize: int):
    x_flat = x.float().contiguous().view(-1)
    numel = x_flat.numel()
    num_blocks = (numel + blocksize - 1) // blocksize
    x_padded = _pad_to_block_multiple(x_flat, blocksize, num_blocks)
    x_blocks = x_padded.view(num_blocks, blocksize)
    absmax = x_blocks.abs().amax(dim=1).clamp_min(1e-12)
    q_blocks = (
        torch.round(x_blocks / absmax.unsqueeze(1) * 127.0)
        .clamp(-127, 127)
        .to(torch.int16)
    )
    q_flat = (q_blocks.view(-1)[:numel] + 128).to(torch.uint8)
    return q_flat.view_as(x), absmax


def _dequantize_blockwise_signed(
    q: torch.Tensor, absmax: torch.Tensor, blocksize: int, shape
) -> torch.Tensor:
    q_flat = q.contiguous().view(-1)
    numel = q_flat.numel()
    num_blocks = absmax.numel()
    q_padded = _pad_to_block_multiple(q_flat, blocksize, num_blocks)
    q_blocks = q_padded.view(num_blocks, blocksize).float()
    out_blocks = ((q_blocks - 128.0) / 127.0) * absmax.unsqueeze(1)
    return out_blocks.view(-1)[:numel].view(shape)


def _quantize_blockwise_unsigned(x: torch.Tensor, blocksize: int):
    x_flat = x.float().contiguous().view(-1)
    numel = x_flat.numel()
    num_blocks = (numel + blocksize - 1) // blocksize
    x_padded = _pad_to_block_multiple(x_flat, blocksize, num_blocks)
    x_blocks = x_padded.view(num_blocks, blocksize)
    absmax = x_blocks.amax(dim=1).clamp_min(1e-12)
    q_blocks = torch.round(x_blocks / absmax.unsqueeze(1) * 255.0).clamp(0, 255)
    q_flat = q_blocks.view(-1)[:numel].to(torch.uint8)
    return q_flat.view_as(x), absmax


def _dequantize_blockwise_unsigned(
    q: torch.Tensor, absmax: torch.Tensor, blocksize: int, shape
) -> torch.Tensor:
    q_flat = q.contiguous().view(-1)
    numel = q_flat.numel()
    num_blocks = absmax.numel()
    q_padded = _pad_to_block_multiple(q_flat, blocksize, num_blocks)
    q_blocks = q_padded.view(num_blocks, blocksize).float()
    out_blocks = (q_blocks / 255.0) * absmax.unsqueeze(1)
    return out_blocks.view(-1)[:numel].view(shape)


class FireFlyOptim(Optimizer):
    def __init__(
        self,
        params,
        lr_int8=0.01,
        lr_dense=1e-3,
        clip_grad=1.0,
        theta=0.0,
        bit_modules=None,
        min_8bit_size=4096,
        block_size=256,
    ):
        defaults = dict(
            lr_int8=float(lr_int8),
            lr_dense=float(lr_dense),
            clip_grad=float(clip_grad),
            theta=float(theta),
            min_8bit_size=int(min_8bit_size),
            block_size=int(block_size),
        )
        super().__init__(params, defaults)
        self.bit_modules = list(bit_modules) if bit_modules is not None else []
        self._bit_handles = (
            [int(m._bit_handle.item()) for m in self.bit_modules]
            if bit_modules is not None
            else []
        )
        self._bit_state: dict[int, dict[str, torch.Tensor | int]] = {}

    def add_bit_modules(self, modules) -> None:
        for module in modules:
            if not isinstance(module, BitLinear):
                raise TypeError(f"expected BitLinear, got {type(module).__name__}")
            if module not in self.bit_modules:
                self.bit_modules.append(module)
                self._bit_handles.append(int(module._bit_handle.item()))

    def _init_8bit_state(self, target: torch.Tensor, block_size: int):
        m_q, m_absmax = _quantize_blockwise_signed(
            torch.zeros_like(target, dtype=torch.float32), block_size
        )
        v_q, v_absmax = _quantize_blockwise_unsigned(
            torch.zeros_like(target, dtype=torch.float32), block_size
        )
        return {
            "m_q": m_q,
            "m_absmax": m_absmax,
            "v_q": v_q,
            "v_absmax": v_absmax,
            "t": 0,
        }

    @torch.no_grad()
    def step(self):
        dense_params = [
            p for g in self.param_groups for p in g["params"] if p.grad is not None
        ]
        if dense_params:
            torch.nn.utils.clip_grad_norm_(
                dense_params, max_norm=self.defaults["clip_grad"]
            )

        beta1, beta2 = 0.9, 0.95
        eps = 1e-8

        for group in self.param_groups:
            lr_dense = float(group.get("lr", group["lr_dense"]))
            min_8bit_size = int(group["min_8bit_size"])
            block_size = int(group["block_size"])

            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad.float().contiguous()
                state = self.state[p]
                use_8bit = p.numel() >= min_8bit_size

                if use_8bit:
                    if "m_q" not in state or state["m_q"].shape != p.shape:
                        state.update(self._init_8bit_state(p, block_size))
                    m = _dequantize_blockwise_signed(
                        state["m_q"], state["m_absmax"], block_size, p.shape
                    )
                    v = _dequantize_blockwise_unsigned(
                        state["v_q"], state["v_absmax"], block_size, p.shape
                    )
                else:
                    if "m" not in state or state["m"].shape != p.shape:
                        state["m"] = torch.zeros_like(p, dtype=torch.float32)
                        state["v"] = torch.zeros_like(p, dtype=torch.float32)
                        state["t"] = 0
                    m = state["m"]
                    v = state["v"]

                state["t"] += 1
                m.mul_(beta1).add_(g, alpha=1 - beta1)
                v.mul_(beta2).addcmul_(g, g, value=1 - beta2)
                m_hat = m / (1 - beta1 ** state["t"])
                v_hat = v / (1 - beta2 ** state["t"])
                p.add_(-lr_dense * (m_hat / (v_hat.sqrt() + eps)).to(dtype=p.dtype))

                if use_8bit:
                    m_q, m_absmax = _quantize_blockwise_signed(m, block_size)
                    v_q, v_absmax = _quantize_blockwise_unsigned(v, block_size)
                    state["m_q"], state["m_absmax"] = m_q, m_absmax
                    state["v_q"], state["v_absmax"] = v_q, v_absmax

        if self.bit_modules:
            cfg = self.param_groups[0]
            lr_int8 = float(cfg["lr_int8"])
            min_8bit_size = int(cfg["min_8bit_size"])
            block_size = int(cfg["block_size"])

            theta = float(cfg.get("theta", 0.0))

            for module, handle in zip(self.bit_modules, self._bit_handles):
                g = module.consume_weight_grad()
                if g is None:
                    continue
                g = g.float().contiguous()
                state = self._bit_state.setdefault(handle, {})
                if "residual" not in state or state["residual"].shape != g.shape:
                    state["residual"] = torch.zeros_like(g, dtype=torch.float32)
                residual = state["residual"]

                use_8bit = g.numel() >= min_8bit_size
                if use_8bit:
                    if "m_q" not in state or state["m_q"].shape != g.shape:
                        state.update(self._init_8bit_state(g, block_size))
                    m = _dequantize_blockwise_signed(
                        state["m_q"], state["m_absmax"], block_size, g.shape
                    )
                    v = _dequantize_blockwise_unsigned(
                        state["v_q"], state["v_absmax"], block_size, g.shape
                    )
                else:
                    if "m" not in state or state["m"].shape != g.shape:
                        state["m"] = torch.zeros_like(g, dtype=torch.float32)
                        state["v"] = torch.zeros_like(g, dtype=torch.float32)
                        state["t"] = 0
                    m = state["m"]
                    v = state["v"]

                state["t"] += 1
                m.mul_(beta1).add_(g, alpha=1.0 - beta1)
                v.mul_(beta2).addcmul_(g, g, value=1.0 - beta2)
                m_hat = m / (1 - beta1 ** state["t"])
                v_hat = v / (1 - beta2 ** state["t"])

                adam_step = -lr_int8 * (m_hat / (v_hat.sqrt() + eps))
                denom = module.weight_scale.float().unsqueeze(1).clamp_min(eps)
                # OU mean reversion: residual pulled toward zero at rate theta
                residual.mul_(1.0 - theta).add_(adam_step / denom)

                abs_res = residual.abs()
                base = torch.floor(abs_res)
                frac = abs_res - base
                extra = (torch.rand_like(frac) < frac).float()
                delta_q_i32 = (torch.sign(residual) * (base + extra)).to(torch.int32)
                if torch.any(delta_q_i32 != 0):
                    update_int8_weight_(module.int_weight, delta_q_i32)
                    residual.sub_(delta_q_i32.float())
                    _invalidate_weight_cache(handle)

                if use_8bit:
                    m_q, m_absmax = _quantize_blockwise_signed(m, block_size)
                    v_q, v_absmax = _quantize_blockwise_unsigned(v, block_size)
                    state["m_q"], state["m_absmax"] = m_q, m_absmax
                    state["v_q"], state["v_absmax"] = v_q, v_absmax

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
        bit_state = state_dict.pop("bit_state", {})
        super().load_state_dict(state_dict)
        self._bit_state = {}
        for handle, per_handle in bit_state.items():
            self._bit_state[int(handle)] = {
                key: value.clone() if torch.is_tensor(value) else value
                for key, value in per_handle.items()
            }
