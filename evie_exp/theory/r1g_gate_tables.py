"""R1g (B1, B2, B4): Figure 1 data and the fidelity ledger.  x = gamma0 sigma^2 in logspace(-3,3,400): s_E, m_0, m_A(w=.85), risk-averse
gain (1+x)^(1/2) vs risk-seeking (1+x)^(-1/2), the B4 view in r (x=(1/w-1) r, r*=1.2143).  Mechanical checks of the closed forms (tol 1e-12):
identity (1-gamma_t s2)(1+gamma0 s2)=1; k_averse*k_seeking=H; m_A(0)=1.0846522890932808; range (1/sqrt(2w), 1/sqrt(w)];
m_0/s_E=(1+x)/sqrt(1+2x)>=1; crossover x*=(1-w)/(2w-1); m_A(r=1)=1/sqrt(w(2-w))=1.01144; clamps (floor 1e-6, cap 8) inactive for w>=1/8;
clip window [3.5294e-5, 1.7647e-2]."""
from __future__ import annotations

import numpy as np

from ..optim import gates as G
from .common import claim, m_0, m_A, main_for, s_E

W = 0.85
# Steps and labels follow the fidelity ledger (Table 2) of the theory notes; labels: Exact / Approximation / Identification / Modelling choice / Design choice.
LEDGER = [
    (1, "Local quadratic loss H x^2/2 near theta*", "Approximation", "assumption: local"),
    (2, "Coordinate-wise model, A=0, B=1, independent coordinates", "Modelling choice", "assumption: local"),
    (3, "H, q constant on the time scale 1/(2 sqrt(Ha))", "Approximation", "remark: locally constant"),
    (4, "Risk-sensitive criterion, sign gamma = -kappa < 0", "Modelling choice", "A1, A6"),
    (5, "Optimal control and stationary gain k* = sqrt(H) (1+kappa q)^(-1/2)", "Exact", "A2, A3, A4"),
    (6, "R = 1", "Exact (w.l.o.g.)", "B1(v)"),
    (7, "Euler step of length eta", "Approximation", "B5: relative error O(eta k0*)"),
    (8, "State replaced by g/H", "Approximation", "B1(ii)"),
    (9, "q = c_q sigma-hat^2", "Identification", "B1(ii), remark on units"),
    (10, "g -> m1-hat (momentum) [design choice];  sqrt(H) -> sqrt(m2-hat) [identification]", "Design choice / Identification", "B1(ii)"),
    (11, "sqrt(1 - gamma_t sigma-hat^2) = (1+gamma0 sigma-hat^2)^(-1/2)", "Exact", "B1(iii)"),
    (12, "Gate floor 1e-6", "Exact (inactive)", "floor lemma"),
    (13, "One-time calibration of gamma0 from w_target", "Design choice", "B4"),
    (14, "Warm-up with gamma0 = 1", "Design choice (breaks B3)", "B3 remark"),
    (15, "Amplify: harmonic kappa_t and factor 1/sqrt(w_target)", "Design choice (exactly described)", "B2"),
    (16, "Global gradient clipping at 5, eps = 1e-8, weight decay", "Design choice", "B3 remark"),
]


def run(rng, smoke, out):
    x = np.logspace(-3, 3, 400)
    r = x / (1 / W - 1)
    sE, m0, mA = s_E(x), m_0(x), m_A(x, W)
    k_av, k_se = np.sqrt(1 + x), (1 + x) ** -0.5
    np.savez(out / "fig_data.npz", x=x, s_E=sE, m_0=m0, m_A=mA, k_averse_over_sqrtH=k_av, k_seeking_over_sqrtH=k_se, r=r, r_star=W / (2 * W - 1),
             averse_feasibility_boundary=np.array([1.0]))
    (out / "ledger.csv").write_text("#,step,label,where\n" + "\n".join(f'{i},"{s}",{l},"{w}"' for i, s, l, w in LEDGER) + "\n")
    tex = ["\\begin{tabular}{rp{7cm}p{3.2cm}p{3cm}}", "\\toprule", "\\# & Step & Label & Where\\\\", "\\midrule"]
    tex += [f"{i} & {s} & {l} & {w}\\\\" for i, s, l, w in LEDGER] + ["\\bottomrule", "\\end{tabular}"]
    (out / "ledger.tex").write_text("\n".join(tex) + "\n")
    s2 = np.logspace(-6, 2, 500); g0 = np.logspace(0, 3.7, 40)
    G0, S2 = np.meshgrid(g0, s2)
    gam_t = G0 / (1 + G0 * S2)
    ident = float(np.max(np.abs((1 - gam_t * S2) * (1 + G0 * S2) - 1) / (1 + G0 * S2)))   # scaled by the cancellation size (1+x)
    prod = float(np.max(np.abs(k_av * k_se - 1)))
    mA0 = float(m_A(0.0, W))
    xs = (1 - W) / (2 * W - 1)
    cross = float(abs(m_A(xs, W) - 1))
    rng_lo, rng_hi = 1 / np.sqrt(2 * W), 1 / np.sqrt(W)
    in_range = bool(np.all(mA > rng_lo) and np.all(mA <= rng_hi + 1e-15))
    ratio_ok = bool(np.all(m0 / sE >= 1 - 1e-15) and np.max(np.abs(m0 / sE - (1 + x) / np.sqrt(1 + 2 * x))) < 1e-12)
    m_at_r1 = float(m_A((1 / W - 1), W))
    pre = (1 + x) / (1 + 2 * x) / W
    clamps = bool(np.all(pre > 1e-6) and np.all(pre < 8.0) and (W >= 1 / 8))
    lo, hi = G.clip_interval(W)
    cl = [
        claim("B1", "shipped gate = exact gain of the risk-seeking branch: identity (1-gamma_t s2)(1+gamma0 s2)=1; k_averse k_seeking = H",
              "max abs error <= 1e-12 (identity error divided by 1+x, the size of the cancellation)", {"identity_max_err_scaled": ident, "gain_product_max_err": prod}, ident <= 1e-12 and prod <= 1e-12),
        claim("B2", "m_A(0) = 1/sqrt(w); range (1/sqrt(2w), 1/sqrt(w)]; m_0/s_E = (1+x)/sqrt(1+2x) >= 1; crossover at x* = (1-w)/(2w-1); both clamps inactive for w >= 1/8",
              "closed forms to 1e-12", {"m_A0": mA0, "m_A0_expected": 1.0846522890932808, "range": [rng_lo, rng_hi], "in_range": in_range, "ratio_ok": ratio_ok,
                                         "crossover_err": cross, "clamps_inactive": clamps},
              abs(mA0 - 1.0846522890932808) < 1e-12 and in_range and ratio_ok and cross < 1e-12 and clamps),
        claim("B4", "calibration lemma: s_E(r=1)^2 = w, m_A(r=1) = 1/sqrt(w(2-w)) = 1.01144, amplifies iff r < r* = w/(2w-1), clip window [3.5294e-5, 1.7647e-2]",
              "closed forms to 1e-12", {"s_E2_at_r1": float(s_E(1 / W - 1) ** 2), "m_A_at_r1": m_at_r1, "r_star": W / (2 * W - 1), "window": [lo, hi]},
              abs(float(s_E(1 / W - 1)) ** 2 - W) < 1e-12 and abs(m_at_r1 - 1 / np.sqrt(W * (2 - W))) < 1e-12 and abs(m_at_r1 - 1.01144) < 1e-5
              and abs(lo - 3.5294e-5) < 1e-8 and abs(hi - 1.7647e-2) < 1e-6),
        claim("ledger", "the fidelity ledger has 16 rows with labels", "16 rows", {"rows": len(LEDGER)}, len(LEDGER) == 16),
    ]
    return {"w": W}, cl, ["fig_data.npz", "ledger.csv", "ledger.tex"]


if __name__ == "__main__":
    raise SystemExit(main_for("R1g", run))
