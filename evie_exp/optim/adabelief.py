"""AdaBelief, own implementation (no dependency), decoupled weight decay.

Matches adabelief_pytorch 0.2.1 with ``weight_decouple=True, rectify=False, amsgrad=False,
fixed_decay=False`` (the reference adds eps *inside* the second-moment EMA, in place):

    p  <- p * (1 - lr*wd)
    m  <- b1 m + (1-b1) g
    s  <- b2 s + (1-b2) (g-m)^2 ;  s <- s + eps
    p  <- p - (lr/bc1) * m / ( sqrt(s)/sqrt(bc2) + eps )

eps defaults to the package default 1e-16 (spec 6.2, decision a); see OPEN_QUESTIONS.md.
"""
from __future__ import annotations

import math

import torch


class AdaBelief(torch.optim.Optimizer):
    def __init__(self, params, lr: float = 1e-3, betas=(0.9, 0.999), eps: float = 1e-16,
                 weight_decay: float = 0.0, rectify: bool = False) -> None:
        if rectify:
            raise NotImplementedError("rectify=True is not part of the plan (OPEN_QUESTIONS.md, AdaBelief rectify)")
        super().__init__(params, dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay))

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr, (b1, b2), eps, wd = group["lr"], group["betas"], group["eps"], group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                if g.is_sparse:
                    raise RuntimeError("AdaBelief does not support sparse gradients")
                if wd > 0.0:
                    p.mul_(1.0 - lr * wd)
                s = self.state[p]
                if "step" not in s:
                    s["step"] = 0
                    s["exp_avg"], s["exp_avg_var"] = torch.zeros_like(p), torch.zeros_like(p)
                s["step"] += 1
                t = s["step"]
                bc1, bc2 = 1.0 - b1 ** t, 1.0 - b2 ** t
                m, v = s["exp_avg"], s["exp_avg_var"]
                m.mul_(b1).add_(g, alpha=1.0 - b1)
                resid = g - m
                v.mul_(b2).addcmul_(resid, resid, value=1.0 - b2).add_(eps)
                denom = (v.sqrt() / math.sqrt(bc2)).add_(eps)
                p.addcdiv_(m, denom, value=-lr / bc1)
        return loss
