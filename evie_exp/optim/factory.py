"""Arm name -> optimizer.  Names are used verbatim in JSON/tables.  Derived arms (M1, M2,
FixedGammaOwn) never get their own tuned LR: the caller passes the LR (and the constant) in."""
from __future__ import annotations

import math
from typing import Dict, Iterable, Optional

import torch

from .adabelief import AdaBelief
from .evie_family import EvieFamily

TUNED_ARMS = ("AdamW", "Evie", "EvieAmplify", "FixedGamma")
DERIVED_ARMS = ("M1", "M2", "FixedGammaOwn")
# Evie-Amplify (EA) component ablations, each removing exactly ONE ingredient of EA = AdamW * m_0(x)/sqrt(w):
EA_TUNED = ("EA_noRecenter",)                                  # no 1/sqrt(w): shrink-only ratio gate m_0(x)      -> "scale" effect
EA_DERIVED = ("M1", "EA_noWarmup", "EA_fixedG0", "EA_fixedSched")  # run at EA's selected LR, no tuning
ALL_ARMS = TUNED_ARMS + DERIVED_ARMS + EA_TUNED + EA_DERIVED[1:] + ("AdaBelief",)


def build_optimizer(arm: str, params: Iterable, lr: float, cfg: dict, *, fixed_gamma: Optional[float] = None,
                    rho_bar: Optional[float] = None, param_types: Optional[Dict[int, str]] = None,
                    diag: bool = True):
    """``cfg`` supplies weight_decay (1e-4) and w_target (0.85). Missing constants raise; nothing is guessed."""
    wd, w = cfg.get("weight_decay", 1e-4), cfg.get("w_target", 0.85)
    common = dict(weight_decay=wd, w_target=w, param_types=param_types, diag=diag,
                  calib_step=cfg.get("calib_step", 125))
    if arm == "AdamW":
        return torch.optim.AdamW(params, lr=lr, weight_decay=wd)
    if arm == "AdamWFamily":  # same code path as the const controls; used by the equivalence test
        return EvieFamily(params, lr, "adamw", **common)
    if arm == "Evie":
        return EvieFamily(params, lr, "evie", **common)
    if arm == "EvieAmplify":
        return EvieFamily(params, lr, "amplify", **common)
    if arm == "EA_noRecenter":      # gate only: sqrt(raw_wp) = m_0(x) in (1/sqrt2, 1]; calibration unchanged (w_target still sets gamma0)
        return EvieFamily(params, lr, "amplify", recenter=False, **common)
    if arm == "EA_noWarmup":        # gamma0 = 0 until calibration -> multiplier exactly 1/sqrt(w) during warm-up (theorem B3 hypothesis H3)
        return EvieFamily(params, lr, "amplify", warmup_gate="off", **common)
    if arm == "EA_fixedG0":         # gamma0 = 1 forever, never calibrated (what ablation.py's fixed settings did)
        return EvieFamily(params, lr, "amplify", calibrate=False, **common)
    if arm == "EA_fixedSched":      # warm-up off, then constant gamma (no harmonic schedule): contrast with EA_noWarmup isolates the schedule
        return EvieFamily(params, lr, "amplify", warmup_gate="off", schedule="fixed", **common)
    if arm in ("FixedGamma", "FixedGammaOwn"):
        if fixed_gamma is None:
            raise ValueError(f"{arm} needs fixed_gamma (gamma_F from calibrate_fixedgamma, or Evie's calib.gamma0)")
        return EvieFamily(params, lr, "fixedgamma", fixed_gamma=float(fixed_gamma), **common)
    if arm == "M1":
        return EvieFamily(params, lr, "const", const_mult=1.0 / math.sqrt(w), **common)
    if arm == "M2":
        if rho_bar is None or not math.isfinite(rho_bar):
            raise ValueError("M2 needs rho_bar (mean realised rho_rms of the Evie eval runs)")
        return EvieFamily(params, lr, "const", const_mult=float(rho_bar), **common)
    if arm == "AdaBelief":
        return AdaBelief(params, lr=lr, weight_decay=wd, eps=cfg.get("adabelief_eps", 1e-16))
    raise ValueError(f"unknown arm {arm!r}")
