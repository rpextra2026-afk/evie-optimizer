"""Pure gate formulas (numpy / torch / python floats). Claim IDs refer to evie_theory.tex."""
from __future__ import annotations

import math

import numpy as np
import torch


def _pow(x, e):
    return x ** e


def s_E(x):
    """Evie gate, x = gamma0 * sigma2.  B1."""
    return _pow(1 + x, -0.5)


def m_0(x):
    """Amplify's saturating exposure factor.  B2."""
    return _pow((1 + x) / (1 + 2 * x), 0.5)


def m_A(x, w):
    """Amplify multiplier m_0(x)/sqrt(w).  B2."""
    return _pow((1 + x) / ((1 + 2 * x) * w), 0.5)


def s_F(y, eps0: float = 1e-6):
    """Fixed-gamma 'difference' gate sqrt(max(1-y, eps0)).  D1."""
    if isinstance(y, torch.Tensor):
        return torch.clamp(1 - y, min=eps0).sqrt()
    if isinstance(y, np.ndarray):
        return np.sqrt(np.maximum(1 - y, eps0))
    return math.sqrt(max(1 - y, eps0))


def clip_interval(w: float) -> tuple:
    """Calibration clip is inactive iff sigma_bar_f^2 lies in this interval.  B4(iv)."""
    return ((1 / w - 1) / 5000.0, (1 / w - 1) / 10.0)


def gamma0_from_sigma(sigma_bar_f2: float, w: float) -> tuple:
    """(clipped gamma0, status in {'lo','hi','inactive'}, raw gamma0).  B4."""
    s = max(float(sigma_bar_f2), 1e-12)
    raw = (1.0 / w - 1.0) / s
    if raw < 10.0:
        return 10.0, "lo", raw
    if raw > 5000.0:
        return 5000.0, "hi", raw
    return raw, "inactive", raw


def crossover_r(w: float) -> float:
    return w / (2 * w - 1)


def zero_noise_factor(w: float) -> float:
    return 1 / math.sqrt(w)
