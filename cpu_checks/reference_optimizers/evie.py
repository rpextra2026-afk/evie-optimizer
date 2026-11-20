"""
evie_optim.optimizer
================================================================================
Evie: risk-sensitive modulation of Adam for stochastic optimization.

Wraps an Adam-style first/second-moment estimate with a shrink-only,
noise-dependent gate derived from a risk-sensitive Riccati equation:

    gamma_t   = gamma0 / (1 + gamma0 * sigma_t^2)
    well_pos  = 1 - gamma_t * sigma_t^2        (in (0, 1])
    update    = -lr * m1_hat * sqrt(well_pos) / sqrt(m2_hat)

gamma0 is auto-calibrated once, at whatever step you call
``calibrate_gamma0()``, from a separate fast-EMA gradient-variance estimate
(tracked independently of the slow EMA used for the actual update), so that
the well-posedness factor sits near ``target_wp`` at that point in training.
Before calibration, the optimizer behaves like plain AdamW.

For the companion optimizer that allows bounded amplification on
low-variance coordinates, see the separate ``evie-amplify-optim`` package.
================================================================================
"""
from __future__ import annotations

import numpy as np
import torch

__all__ = ["EvieOptimizer"]

_DEFAULT_TARGET_WP = 0.85
_DEFAULT_FAST_BETA = 0.9
_GAMMA0_MIN = 10.0
_GAMMA0_MAX = 5000.0


class EvieOptimizer(torch.optim.Optimizer):
    """Base Evie optimizer (shrink-only risk-sensitive gate on top of Adam moments).

    Call ``calibrate_gamma0()`` once, partway through training (the paper
    uses a step between 100 and 150), to set ``gamma0`` from the observed
    gradient variance. Before calibration, the optimizer behaves like plain
    AdamW.

    Args:
        params: iterable of parameters to optimize.
        lr: learning rate.
        gamma0: initial risk-aversion ceiling. Overwritten by
            ``calibrate_gamma0()`` if/when that is called.
        target_wp: target well-posedness factor used during calibration.
        beta1, beta2: Adam moment decay rates.
        eps: numerical floor for the second-moment denominator.
        weight_decay: decoupled weight decay coefficient (AdamW-style).
        fast_beta: decay rate for the separate fast-EMA variance estimate
            used only for gamma0 calibration.
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        gamma0: float = 1.0,
        target_wp: float = _DEFAULT_TARGET_WP,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        fast_beta: float = _DEFAULT_FAST_BETA,
    ):
        defaults = dict(
            lr=lr,
            gamma0=gamma0,
            target_wp=target_wp,
            beta1=beta1,
            beta2=beta2,
            eps=eps,
            weight_decay=weight_decay,
            calibrated=False,
            fast_beta=fast_beta,
        )
        super().__init__(params, defaults)

    def calibrate_gamma0(self) -> None:
        """Set gamma0 from the current fast-EMA gradient-variance estimate.

        No-op if there is no optimizer state yet (i.e. ``step()`` has not
        been called), or if a group has already been calibrated.
        """
        for group in self.param_groups:
            if group["calibrated"]:
                continue
            target = group["target_wp"]
            sigma2_list = []
            for p in group["params"]:
                s = self.state.get(p, {})
                if "fast_m1" not in s:
                    continue
                sigma2 = (s["fast_m2"] - s["fast_m1"].pow(2)).clamp_(min=0.0)
                sigma2_list.append(sigma2.mean().item())
            if not sigma2_list:
                continue
            mean_s2 = max(float(np.mean(sigma2_list)), 1e-12)
            new_g0 = float(np.clip((1.0 / target - 1.0) / mean_s2, _GAMMA0_MIN, _GAMMA0_MAX))
            group["gamma0"] = new_g0
            group["calibrated"] = True

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr, g0, b1, b2, eps, wd = (
                group["lr"], group["gamma0"], group["beta1"],
                group["beta2"], group["eps"], group["weight_decay"],
            )
            fb = group["fast_beta"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                if wd > 0.0:
                    p.data.mul_(1.0 - lr * wd)
                grad = p.grad.data
                s = self.state[p]
                if "step" not in s:
                    s["step"] = 0
                    s["m1"] = torch.zeros_like(p.data)
                    s["m2"] = torch.zeros_like(p.data)
                    s["fast_m1"] = torch.zeros_like(p.data)
                    s["fast_m2"] = torch.zeros_like(p.data)
                s["step"] += 1
                t = s["step"]

                # main (slow) EMA used for the actual update
                s["m1"].mul_(b1).add_(grad, alpha=1.0 - b1)
                s["m2"].mul_(b2).addcmul_(grad, grad, value=1.0 - b2)
                # separate fast EMA used only for gamma0 calibration
                s["fast_m1"].mul_(fb).add_(grad, alpha=1.0 - fb)
                s["fast_m2"].mul_(fb).addcmul_(grad, grad, value=1.0 - fb)

                m1c = s["m1"] / (1.0 - b1 ** t)
                m2c = s["m2"] / (1.0 - b2 ** t)
                sigma2 = (m2c - m1c.pow(2)).clamp_(min=0.0)
                gamma_t = g0 / (1.0 + g0 * sigma2)
                well_pos = (1.0 - gamma_t * sigma2).clamp_(min=1e-6)
                denom = m2c.sqrt().clamp_(min=eps)
                p.data.addcdiv_(m1c.mul(well_pos.sqrt()), denom, value=-lr)
        return loss
