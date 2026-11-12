"""R1c (C0, C1): SGD on gated diagonal quadratics vs the C1 recursions; opposite noise floors.
PASS (C1): z=|emp-closed|/SE has max z < 5 over (coordinate, n, xbar) for E x_n^2 and E x_n; stability edge: bounded for
theta<2 (E x^2 within 5 SE of V_inf), divergent for theta>2; 1-rho = 2 theta + O(theta^2), V_inf = theta sigma^2/(2h^2)+O(theta^2) (rel err <= theta, theta<=0.3).
PASS (C0): for every coordinate the control-model variance q/(2k*) is strictly increasing in kappa while the SGD floor is strictly decreasing;
the simulated floor (coordinate mean of E x_N^2 / V_inf(xbar=0)) decreases strictly over the four xbar levels."""
from __future__ import annotations

import numpy as np

from .common import claim, main_for, s_E


def run(rng, smoke, out):
    d, N, M = (64, 2000, 20000) if not smoke else (16, 300, 1500)
    z = rng.standard_normal(d)
    zp = rng.standard_normal(d)
    h, sig = np.exp(0.5 * z), np.exp(0.75 * zp)
    x0 = 1.0
    v = h ** 2 * x0 ** 2 + sig ** 2
    eta = 0.3 / np.max(h / np.sqrt(v))
    xbars = [0.0, 0.1765, 1.0, 5.0]
    ns = np.unique(np.round(np.logspace(0, np.log10(N), 25)).astype(int))
    maxz, rows = 0.0, []
    floors = []
    V_inf0 = None
    for xb in xbars:
        g0 = xb / np.mean(sig ** 2)
        m = s_E(g0 * sig ** 2)
        th = eta * m * h / np.sqrt(v)
        rho = (1 - th) ** 2
        Vinf = th * sig ** 2 / (h ** 2 * (2 - th))
        if xb == 0.0:
            V_inf0 = Vinf.copy()
        x = np.full((M, d), x0)
        sc = eta * m / np.sqrt(v)
        sums = {}
        for n in range(1, N + 1):
            x = x - sc * (h * x + sig * rng.standard_normal((M, d)))
            if n in ns:
                sums[n] = (x.mean(axis=0), (x ** 2).mean(axis=0), x.std(axis=0, ddof=1), (x ** 2).std(axis=0, ddof=1))
        for n in ns:
            mean_e, m2_e, sd, sd2 = sums[n]
            cl2 = rho ** n * x0 ** 2 + Vinf * (1 - rho ** n)
            cl1 = (1 - th) ** n * x0
            z2 = np.abs(m2_e - cl2) / (sd2 / np.sqrt(M))
            z1 = np.abs(mean_e - cl1) / (sd / np.sqrt(M))
            maxz = max(maxz, float(z1.max()), float(z2.max()))
        floors.append(float(np.mean(sums[ns[-1]][1] / V_inf0)))
        rows.append(f"{xb},{eta},{float(np.mean(th))},{floors[-1]}")
    # stability edge (unit h, sigma)
    Ms, Ns = (20000, 200) if not smoke else (2000, 120)
    edge, edge_ok = [], True
    for th in (1.0, 1.9, 2.1):
        x = np.full(Ms, 1.0)
        for _ in range(Ns):
            x = (1 - th) * x - th * rng.standard_normal(Ms)
        e2, se = float((x ** 2).mean()), float((x ** 2).std(ddof=1) / np.sqrt(Ms))
        if th < 2:
            Vi = th / (2 - th)
            ok = abs(e2 - Vi) <= 5 * se
        else:
            ok = e2 > 1e3
        edge_ok &= ok
        edge.append({"theta": th, "E_x2": e2, "SE": se, "V_inf_closed": th / (2 - th) if th < 2 else None, "ok": bool(ok)})
    claims = [claim("C1", "E x_n^2 = rho^n x0^2 + V_inf (1-rho^n), E x_n = (1-theta)^n x0, stable iff 0<theta<2",
                    "max z<5 over coordinates x n x xbar (both moments); edge theta in {1,1.9,2.1}",
                    {"max_z": maxz, "coords": d, "n_points": len(ns), "replicates": M, "edge": edge}, maxz < 5 and edge_ok)]
    ths = np.linspace(0.005, 0.3, 60)
    e_rho = np.abs((1 - (1 - ths) ** 2) - 2 * ths) / (2 * ths)
    e_V = np.abs(ths / (2 - ths) - ths / 2) / (ths / 2)
    claims.append(claim("C1-asym", "1-rho = 2 theta + O(theta^2); V_inf = theta sigma^2/(2h^2) + O(theta^2)", "relative error <= theta for theta <= 0.3",
                        {"max_ratio_rho": float(np.max(e_rho / ths)), "max_ratio_V": float(np.max(e_V / ths))}, bool(np.all(e_rho <= ths) and np.all(e_V <= ths))))
    # C0
    kap = np.concatenate([[0.0], np.logspace(-2, 2, 11)]) / np.mean(sig ** 2)
    q_i = sig ** 2
    ctrl = np.stack([q_i / (2 * np.sqrt(h) * (1 + k * q_i) ** -0.5) for k in kap])           # [K, d]; k* = sqrt(H)(1+kappa q)^(-1/2), H=h
    th0 = eta * h / np.sqrt(v)
    th_k = th0[None, :] * s_E(kap[:, None] * sig[None, :] ** 2)
    sgd = th_k * sig ** 2 / (h ** 2 * (2 - th_k))
    up = bool(np.all(np.diff(ctrl, axis=0) > 0))
    down = bool(np.all(np.diff(sgd, axis=0) < 0))
    emp_dec = bool(np.all(np.diff(floors) < 0))
    np.savez(out / "fig_data.npz", kappa_norm=kap * np.mean(sig ** 2), ctrl_var_norm=(ctrl / ctrl[0]).mean(axis=1), sgd_floor_norm=(sgd / sgd[0]).mean(axis=1),
             xbar=np.array(xbars), emp_floor_norm=np.array(floors))
    (out / "table.csv").write_text("xbar,eta,mean_theta,empirical_floor_over_floor_at_xbar0\n" + "\n".join(rows) + "\n")
    claims.append(claim("C0", "additive control model: variance increases with the gate strength kappa; SGD model: the noise floor decreases",
                        "strictly opposite for every coordinate; empirical floors decrease over the four xbar levels",
                        {"control_increasing_all_coords": up, "sgd_decreasing_all_coords": down, "empirical_floor_ratios": floors}, up and down and emp_dec))
    return {"d": d, "N": N, "M": M, "eta": float(eta), "xbars": xbars}, claims, ["table.csv", "fig_data.npz"]


if __name__ == "__main__":
    raise SystemExit(main_for("R1c", run))
