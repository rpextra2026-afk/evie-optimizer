"""R1f (C3): bound for SGD with a predictable diagonal preconditioner D_t (delayed statistics, clipped into [d_min,d_max]).
F(theta)=sum_i h_i theta_i^2/2 + 0.1 cos theta_i, d=20, oracle g = grad F + xi, xi~N(0,sigma_c^2 I), sigma^2 = d sigma_c^2, T=3000, 200 runs/setting.
PASS: (1/T) sum E||grad F||^2 (mean over runs, upper 95 % = mean+2SE) <= 2 Delta0/(eta d_min T) + L eta d_max^2 sigma^2/d_min in every setting,
for Delta0 = F(theta0)-F* and for the loose F_lb = -0.1 d; tuned-eta bound 2L Delta0 kD^2/T + 2 kD sqrt(2 L Delta0 sigma^2/T);
uniform scaling D->cD, eta->eta/c gives identical iterates (<1e-12). FALSIFIED if any setting exceeds the bound by more than 4 SE."""
from __future__ import annotations

import numpy as np

from .common import claim, main_for, s_E


def grad(th, h):
    return h * th - 0.1 * np.sin(th)


def F(th, h):
    return (0.5 * h * th ** 2 + 0.1 * np.cos(th)).sum(axis=-1)


def precond(variant, vbar, sig2_hat, g0, w=0.85):
    p = 1.0 / np.maximum(np.sqrt(vbar), 1e-8)
    if variant == "plain":
        s = 1.0
    elif variant == "evie":
        s = s_E(g0 * sig2_hat)
    elif variant == "amplify":
        s = np.sqrt((1 + g0 * sig2_hat) / (1 + 2 * g0 * sig2_hat)) / np.sqrt(w)      # m_A
    elif variant == "misaligned":
        s = (1.0 + g0 * sig2_hat) ** 0.5
    else:
        raise ValueError(variant)
    return s * p


def simulate(variant, theta0, h, sigma_c, eta, T, runs, rng, bounds=None, scale=1.0, noise=None):
    """Returns (mean_grad2_per_run [runs], d_min, d_max, final theta). D_t built from statistics of steps < t only."""
    d = theta0.size
    th = np.tile(theta0, (runs, 1))
    m1 = np.zeros((runs, d)); m2 = np.zeros((runs, d)); vb = np.zeros((runs, d))
    g0, step = 1.0, 0
    acc = np.zeros(runs)
    dmin, dmax = np.inf, 0.0
    fm1 = np.zeros((runs, d)); fm2 = np.zeros((runs, d))
    prev_g = None
    for t in range(T):
        if prev_g is not None:                                   # statistics from steps < t (predictable)
            step += 1
            m1 = 0.9 * m1 + 0.1 * prev_g
            m2 = 0.99 * m2 + 0.01 * prev_g ** 2
            vb = m2 / (1 - 0.99 ** step)
            fm1 = 0.9 * fm1 + 0.1 * prev_g
            fm2 = 0.9 * fm2 + 0.1 * prev_g ** 2
        m1c = m1 / (1 - 0.9 ** max(step, 1))
        sig2 = np.maximum(vb - m1c ** 2, 0.0)
        if step == 125:                                          # one-time calibration, as in the optimiser
            sb = float(np.mean(np.maximum(fm2 - fm1 ** 2, 0.0)))
            g0 = float(np.clip((1 / 0.85 - 1) / max(sb, 1e-12), 10.0, 5000.0))
        D = precond(variant, vb if step > 0 else np.ones_like(vb), sig2, g0)
        if bounds is not None:
            D = np.clip(D, bounds[0], bounds[1])
        D = D * scale
        dmin, dmax = min(dmin, float(D.min())), max(dmax, float(D.max()))
        gt = grad(th, h)
        acc += (gt ** 2).sum(axis=1)
        xi = sigma_c * (noise[t] if noise is not None else rng.standard_normal(th.shape))
        g = gt + xi
        th = th - eta * D * g
        prev_g = g
    return acc / T, dmin, dmax, th


def run(rng, smoke, out):
    d, T, runs = (20, 3000, 200) if not smoke else (20, 400, 20)
    h = np.linspace(0.5, 4.0, d)
    L = h.max() + 0.1
    theta0 = rng.uniform(-3.0, 3.0, d)
    Fstar = 0.1 * d                                              # theta=0 is the unique stationary point for h>0.1
    Fcheck = F(np.zeros(d), h)
    D0 = {"tight": F(theta0, h) - Fstar, "loose": F(theta0, h) + 0.1 * d}
    variants = ["plain", "evie", "amplify", "misaligned"]
    rows, worst, all_ok, slack_min = [], -np.inf, True, np.inf
    prelim = {}
    for sc in (0.05, 0.2, 1.0):
        for v in variants:
            _, a, b, _ = simulate(v, theta0, h, sc, 1e-3, T, 20, rng)
            prelim[(v, sc)] = (a / 1.05, b * 1.05)
    for sc in (0.05, 0.2, 1.0):
        sig2 = d * sc ** 2
        for v in variants:
            dmin, dmax = prelim[(v, sc)]
            kD = dmax / dmin
            eta_max = dmin / (L * dmax ** 2)
            etas = [("0.25", 0.25 * eta_max), ("0.5", 0.5 * eta_max), ("1", eta_max)]
            for lab_D, D0v in D0.items():
                eta_t = min(eta_max, np.sqrt(2 * D0v / (T * L * sig2)) / dmax)
                etas_run = etas + [("tuned", eta_t)]
                for lab, eta in etas_run:
                    g2, _, _, _ = simulate(v, theta0, h, sc, eta, T, runs, rng, bounds=(dmin, dmax))
                    mean, se = float(g2.mean()), float(g2.std(ddof=1) / np.sqrt(runs))
                    bound = 2 * D0v / (eta * dmin * T) + L * eta * dmax ** 2 * sig2 / dmin
                    over = (mean + 2 * se) - bound
                    worst = max(worst, (mean - bound) / max(se, 1e-300))
                    all_ok &= (mean - bound) <= 4 * se
                    slack_min = min(slack_min, bound / mean)
                    if lab == "tuned":
                        tb = 2 * L * D0v * kD ** 2 / T + 2 * kD * np.sqrt(2 * L * D0v * sig2 / T)
                        all_ok &= (mean - tb) <= 4 * se
                    rows.append(f"{v},{sc},{lab_D},{lab},{eta},{dmin},{dmax},{kD},{mean},{se},{bound},{bound / mean},{int(over <= 0)}")
    (out / "table.csv").write_text("variant,sigma_c,Delta0,eta_rule,eta,d_min,d_max,kappa_D,mean_grad2,SE,bound,slack_ratio,upper95_below_bound\n" + "\n".join(rows) + "\n")
    # uniform-scaling invariance of the update D -> cD, eta -> eta/c
    nz = rng.standard_normal((200, 1, d))
    a = simulate("evie", theta0, h, 0.2, 1e-3, 200, 1, rng, scale=1.0, noise=nz)[3]
    b = simulate("evie", theta0, h, 0.2, 1e-3 / 3.0, 200, 1, rng, scale=3.0, noise=nz)[3]
    inv = float(np.max(np.abs(a - b)))
    np.savez(out / "fig_data.npz", slack=np.array([float(r.split(",")[11]) for r in rows]))
    cl = [claim("C3", "the preconditioned SGD bound (1/T) sum E||grad F||^2 <= 2 Delta0/(eta d_min T) + L eta d_max^2 sigma^2/d_min holds with predictable clipped D_t",
                "mean-bound <= 4 SE in every setting (both Delta0, incl. tuned-eta bound); uniform-scaling invariance < 1e-12",
                {"settings": len(rows), "max_z_over_bound": float(worst), "min_slack_ratio_bound_over_mean": float(slack_min), "scaling_invariance_max_abs": inv,
                 "F_star_check": float(Fcheck), "note": "D_t clipped into [d_min/1.05.., 1.05 d_max] from a preliminary run is part of the tested algorithm"},
                bool(all_ok) and inv < 1e-12)]
    return {"d": d, "T": T, "runs": runs, "L": float(L)}, cl, ["table.csv", "fig_data.npz"]


if __name__ == "__main__":
    raise SystemExit(main_for("R1f", run))
