"""EvieFamily: one optimizer class, many arms.  Semantics mirror the shipped Evie/Evie-Amplify
classes exactly (per-tensor loop, decoupled WD with the scheduler-updated lr, eps clamp on
sqrt(m2_hat), bias-corrected moments, calibration called by the loop *after* step()).

Modes: adamw | evie | amplify | const | fixedgamma.   Claim map: see docs / spec 7.2, 9.3.
"""
from __future__ import annotations

import math
from typing import Dict, Optional

import numpy as np
import torch

from .gates import gamma0_from_sigma

MODES = ("adamw", "evie", "amplify", "const", "fixedgamma")
TYPES = ("conv_w", "conv_b", "linear_w", "linear_b", "embed", "other")
_N = 9  # accumulator slots: n, sum_g, sum_g2, sum_(g*u)^2, sum_u^2, n_floor, n_cap, n_expo, n_amp


def classify_param(name: str, p: torch.Tensor) -> str:
    n = name.lower()
    if "emb" in n:
        return "embed"
    if p.dim() == 4:
        return "conv_w"
    if "conv" in n and p.dim() == 1:
        return "conv_b"
    if p.dim() == 2:
        return "linear_w"
    if p.dim() == 1:
        return "linear_b"
    return "other"


class EvieFamily(torch.optim.Optimizer):
    def __init__(self, params, lr: float, mode: str, weight_decay: float = 1e-4, beta1: float = 0.9,
                 beta2: float = 0.999, eps: float = 1e-8, fast_beta: float = 0.9, w_target: float = 0.85,
                 gamma0_warmup: float = 1.0, warmup_gate: str = "gamma0", calib_step: int = 125,
                 well_pos_max: float = 8.0, floor: float = 1e-6, const_mult: float = 1.0,
                 fixed_gamma: Optional[float] = None, schedule: str = "harmonic", recenter: bool = True,
                 calibrate: bool = True, diag: bool = True,
                 param_types: Optional[Dict[int, str]] = None, hi_sample: int = 200_000) -> None:
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
        if warmup_gate not in ("gamma0", "off"):
            raise ValueError("warmup_gate must be 'gamma0' or 'off'")
        if schedule not in ("harmonic", "fixed"):
            raise ValueError("schedule must be 'harmonic' (shipped: gamma_t = g0/(1+g0*s2)) or 'fixed' (gamma_t = g0)")
        if mode == "fixedgamma" and (fixed_gamma is None or not math.isfinite(fixed_gamma) or fixed_gamma < 0):
            raise ValueError("fixedgamma mode needs a finite fixed_gamma >= 0 (never guessed)")
        defaults = dict(lr=lr, weight_decay=weight_decay, beta1=beta1, beta2=beta2, eps=eps, fast_beta=fast_beta)
        super().__init__(params, defaults)
        self.mode, self.w_target, self.floor, self.wp_max = mode, w_target, floor, well_pos_max
        self.const_mult, self.fixed_gamma = const_mult, fixed_gamma
        self.schedule, self.recenter, self.calibrate_enabled = schedule, recenter, calibrate
        self.calib_step, self.diag = calib_step, diag
        self.gamma0 = 0.0 if warmup_gate == "off" else float(gamma0_warmup)
        self.calibrated = False
        self.calib: dict = {"sigma_bar_f2": None, "gamma0_raw": None, "gamma0": None, "clip_status": "na",
                            "step": calib_step}
        self.param_types = param_types or {}
        self._uses_fast = mode in ("evie", "amplify") and calibrate
        if mode in ("evie", "amplify") and not calibrate:
            self.calib = {"sigma_bar_f2": None, "gamma0_raw": None, "gamma0": self.gamma0, "clip_status": "na", "step": None}
        self._acc: Dict[str, torch.Tensor] = {}
        self._hi_sample, self._idx_cache = hi_sample, {}
        self._clip = {"n": 0}

    # ------------------------------------------------------------------ calibration
    def calibrate_gamma0(self) -> None:
        """B4. Mean over tensors of mean(max(f2-f1^2,0)); floor 1e-12; clip [10,5000]. Syncs once."""
        if self.calibrated or not self._uses_fast:
            return
        vals = []
        for group in self.param_groups:
            for p in group["params"]:
                s = self.state.get(p, {})
                if "f1" in s:
                    vals.append((s["f2"] - s["f1"].pow(2)).clamp_(min=0.0).mean().item())
        if not vals:
            return
        mean_s2 = max(float(np.mean(vals)), 1e-12)
        g0, status, raw = gamma0_from_sigma(mean_s2, self.w_target)
        self.gamma0, self.calibrated = float(g0), True
        self.calib = {"sigma_bar_f2": mean_s2, "gamma0_raw": float(raw), "gamma0": float(g0),
                      "clip_status": status, "step": self.calib_step}

    # ------------------------------------------------------------------ step
    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        g0, mode = self.gamma0, self.mode
        for group in self.param_groups:
            lr, wd, b1, b2 = group["lr"], group["weight_decay"], group["beta1"], group["beta2"]
            eps, fb = group["eps"], group["fast_beta"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                if wd > 0.0:
                    p.mul_(1.0 - lr * wd)
                g = p.grad
                s = self.state[p]
                if "step" not in s:
                    s["step"] = 0
                    s["m1"], s["m2"] = torch.zeros_like(p), torch.zeros_like(p)
                    if self._uses_fast:
                        s["f1"], s["f2"] = torch.zeros_like(p), torch.zeros_like(p)
                s["step"] += 1
                t = s["step"]
                s["m1"].mul_(b1).add_(g, alpha=1.0 - b1)
                s["m2"].mul_(b2).addcmul_(g, g, value=1.0 - b2)
                if self._uses_fast:
                    s["f1"].mul_(fb).add_(g, alpha=1.0 - fb)
                    s["f2"].mul_(fb).addcmul_(g, g, value=1.0 - fb)
                m1c = s["m1"] / (1.0 - b1 ** t)
                m2c = s["m2"] / (1.0 - b2 ** t)
                denom = m2c.sqrt().clamp_(min=eps)
                pre = None  # pre-clamp well-posedness factor (for floor/cap diagnostics)
                x = None    # exposure gamma*sigma2 (for expo diagnostics)
                if mode == "adamw":
                    mult = 1.0
                elif mode == "const":
                    mult = self.const_mult
                else:
                    sigma2 = (m2c - m1c.pow(2)).clamp_(min=0.0)
                    if mode == "fixedgamma":
                        x = self.fixed_gamma * sigma2
                        pre = 1.0 - x
                        well_pos = pre.clamp(min=self.floor)
                    else:
                        gamma_t = g0 / (1.0 + g0 * sigma2) if self.schedule == "harmonic" else g0
                        x = g0 * sigma2
                        if mode == "evie":
                            pre = 1.0 - gamma_t * sigma2
                            well_pos = pre.clamp(min=self.floor)
                        else:  # amplify
                            pre = (1.0 / (1.0 + gamma_t * sigma2)) / (self.w_target if self.recenter else 1.0)
                            well_pos = pre.clamp(min=self.floor, max=self.wp_max)
                    mult = well_pos.sqrt()
                if self.diag:
                    self._accumulate(p, m1c, denom, mult, pre, x)
                p.addcdiv_(m1c.mul(mult), denom, value=-lr)
        return loss

    # ------------------------------------------------------------------ diagnostics
    def _accumulate(self, p, m1c, denom, mult, pre, x) -> None:
        typ = self.param_types.get(id(p), "other")
        acc = self._acc.get(typ)
        if acc is None:
            acc = self._acc[typ] = torch.zeros(_N, device=p.device, dtype=torch.float64)
        u = m1c / denom
        u2 = u.pow(2).sum()
        n = float(p.numel())
        if isinstance(mult, float):
            gsum, g2sum, gu2 = mult * n, mult * mult * n, mult * mult * u2
            n_amp = n if mult > 1.0 else 0.0
            zero = torch.zeros((), device=p.device)
            vec = [torch.tensor(n, device=p.device), torch.as_tensor(gsum, device=p.device),
                   torch.as_tensor(g2sum, device=p.device), gu2, u2, zero, zero, zero,
                   torch.as_tensor(n_amp, device=p.device)]
        else:
            gu2 = (mult * u).pow(2).sum()
            n_floor = (pre <= self.floor).sum() if pre is not None else torch.zeros((), device=p.device)
            n_cap = (pre >= self.wp_max).sum() if (self.mode == "amplify") else torch.zeros((), device=p.device)
            n_exp = (x > 1.0).sum() if x is not None else torch.zeros((), device=p.device)
            vec = [torch.tensor(n, device=p.device), mult.sum(), mult.pow(2).sum(), gu2, u2, n_floor, n_cap, n_exp,
                   (mult > 1.0).sum()]
        acc += torch.stack([torch.as_tensor(v, device=p.device, dtype=torch.float64).reshape(()) for v in vec])

    @staticmethod
    def _summ(a: np.ndarray) -> dict:
        n = max(a[0], 1.0)
        return {"gate_mean": a[1] / n, "gate_sq_mean": a[2] / n, "rho_mean": a[1] / n,
                "rho_rms": math.sqrt(a[3] / a[4]) if a[4] > 0 else float("nan"),
                "floor_frac": a[5] / n, "cap_frac": a[6] / n, "expo_gt1_frac": a[7] / n, "amplify_frac": a[8] / n}

    def _flat_state(self):
        ms1, ms2 = [], []
        for group in self.param_groups:
            b1, b2 = group["beta1"], group["beta2"]
            for p in group["params"]:
                s = self.state.get(p, {})
                if "m1" in s:
                    t = s["step"]
                    ms1.append((s["m1"] / (1 - b1 ** t)).flatten())
                    ms2.append((s["m2"] / (1 - b2 ** t)).flatten())
        if not ms1:
            return None, None
        return torch.cat(ms1), torch.cat(ms2)

    def heterogeneity(self) -> dict:
        """hi_cv, psi_logstd, psi_gate_spearman and r-quantiles at the current state (descriptive)."""
        from scipy.stats import spearmanr
        m1c, m2c = self._flat_state()
        out = {"hi_cv": float("nan"), "psi_logstd": float("nan"), "psi_gate_spearman": float("nan"), "r": None}
        if m1c is None:
            return out
        sig2 = (m2c - m1c.pow(2)).clamp(min=0.0)
        mask = sig2 > 1e-30
        if int(mask.sum()) < 10:
            return out
        sig2, m2 = sig2[mask].double(), m2c[mask].double()
        ratio = sig2 / m2
        out["hi_cv"] = float(ratio.std() / ratio.mean())
        logpsi = torch.log(m2 / sig2)
        out["psi_logstd"] = float(logpsi.std())
        n = sig2.numel()
        if n not in self._idx_cache:
            g = torch.Generator().manual_seed(0)
            self._idx_cache[n] = torch.randperm(n, generator=g)[: self._hi_sample].to(sig2.device)
        idx = self._idx_cache[n]
        lg = (-0.5 * torch.log1p(max(self.gamma0, 1e-12) * sig2))[idx].cpu().numpy()
        lp = logpsi[idx].cpu().numpy()
        if np.ptp(lg) > 0 and np.ptp(lp) > 0:
            out["psi_gate_spearman"] = float(spearmanr(lg, lp).statistic)
        if self.calibrated and self.calib["sigma_bar_f2"]:
            r = (sig2[idx] / self.calib["sigma_bar_f2"]).cpu().numpy()
            qs = np.quantile(r, [0.01, 0.10, 0.50, 0.90, 0.99, 0.999])
            out["r"] = dict(zip(["q01", "q10", "q50", "q90", "q99", "q999"], map(float, qs)))
        return out

    def pop_diagnostics(self) -> dict:
        """Interval diagnostics since the last pop (one host sync), then reset."""
        res = {"by_type": {}, "global": None}
        tot = np.zeros(_N)
        for typ, acc in self._acc.items():
            a = acc.cpu().numpy()
            res["by_type"][typ] = self._summ(a)
            tot += a
            acc.zero_()
        if tot[0] > 0:
            res["global"] = self._summ(tot)
        res["het"] = self.heterogeneity() if self.diag else {}
        return res
