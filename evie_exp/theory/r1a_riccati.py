"""R1a (A3): Riccati ODE dS/dtau = H - a S^2, S(0)=0, vs closed forms; offset c' = (q/2) S; escape time; bound.
PASS: max rel err of S < 1e-8 per case; escape-time error < 1e-6 tau_esc; 0 < S*-S <= 2 S* exp(-2 sqrt(Ha) tau) (tol 1e-12)
and S strictly increasing; c closed form max rel err < 1e-8."""
from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp

from .common import claim, main_for


def closed_S(a, H, tau):
    if a > 0:
        return np.sqrt(H / a) * np.tanh(np.sqrt(H * a) * tau)
    if a == 0:
        return H * tau
    return np.sqrt(H / -a) * np.tan(np.sqrt(H * -a) * tau)


def closed_c(a, H, q, tau):
    if a > 0:
        return q / (2 * a) * np.log(np.cosh(np.sqrt(H * a) * tau))
    if a == 0:
        return q * H * tau ** 2 / 4
    return -q / (2 * -a) * np.log(np.cos(np.sqrt(H * -a) * tau))


def run(rng, smoke, out):
    A = [-0.5, 0.0, 0.5, 2.0]
    Hs = [0.25, 1.0, 4.0, 16.0]
    Qs = [0.5, 2.0]
    if smoke:
        Hs, Qs = [1.0], [2.0]
    worst_S = worst_c = worst_esc = 0.0
    bound_viol, mono_viol, n_cases = 0.0, 0, 0
    rows = []
    for a in A:
        for H in Hs:
            for q in Qs:
                gamma = (1 - a) / q
                tau_esc = np.pi / (2 * np.sqrt(H * -a)) if a < 0 else np.inf
                T = 0.98 * tau_esc if a < 0 else 6 / np.sqrt(H * max(abs(a), 0.5))
                f = lambda t, y: [H - a * y[0] ** 2, 0.5 * q * y[0]]
                sol = solve_ivp(f, [0, T], [0.0, 0.0], method="DOP853", rtol=1e-12, atol=1e-14, dense_output=True)
                tau = np.linspace(T / 400, T, 400)
                S_num, c_num = sol.sol(tau)
                S_cf, c_cf = closed_S(a, H, tau), closed_c(a, H, q, tau)
                eS = float(np.max(np.abs(S_num - S_cf) / np.abs(S_cf)))
                ec = float(np.max(np.abs(c_num - c_cf) / np.abs(c_cf)))
                worst_S, worst_c = max(worst_S, eS), max(worst_c, ec)
                if np.any(np.diff(S_num) <= 0):
                    mono_viol += 1
                if a > 0:
                    Sst = np.sqrt(H / a)
                    gap = Sst - S_num
                    ub = 2 * Sst * np.exp(-2 * np.sqrt(H * a) * tau)
                    bound_viol = max(bound_viol, float(np.max(gap - ub)) - 1e-12 if np.max(gap - ub) > 1e-12 else 0.0, 1.0 if np.any(gap <= 0) else 0.0)
                if a < 0:
                    ev = lambda t, y: y[0] - 1e8
                    ev.terminal = True
                    s2 = solve_ivp(f, [0, tau_esc], [0.0, 0.0], method="DOP853", rtol=1e-12, atol=1e-14, events=ev)
                    t_ev = float(s2.t_events[0][0]) if len(s2.t_events[0]) else np.nan
                    e_esc = abs(t_ev - tau_esc) / tau_esc
                    worst_esc = max(worst_esc, e_esc if np.isfinite(e_esc) else 1.0)
                n_cases += 1
                rows.append(f"{a},{H},{q},{gamma},{T},{eS},{ec}")
    (out / "table.csv").write_text("a,H,q,gamma,T,max_rel_err_S,max_rel_err_c\n" + "\n".join(rows) + "\n")
    cl = [claim("A3", "Riccati solution S and offset c equal their closed forms; S increasing; escape time pi/(2 sqrt(H|a|)); two-sided bound",
                "max rel err S<1e-8, c<1e-8; escape err<1e-6 tau_esc; bound violation<=1e-12; S strictly increasing",
                {"cases": n_cases, "max_rel_err_S": worst_S, "max_rel_err_c": worst_c, "max_rel_err_escape": worst_esc,
                 "bound_violation": bound_viol, "monotonicity_violations": mono_viol},
                worst_S < 1e-8 and worst_c < 1e-8 and worst_esc < 1e-6 and bound_viol <= 1e-12 and mono_viol == 0)]
    return {"grid": "a x H x q", "n_cases": n_cases}, cl, ["table.csv"]


if __name__ == "__main__":
    raise SystemExit(main_for("R1a", run))
