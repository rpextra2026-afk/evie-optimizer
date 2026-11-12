"""R1d (C2): heterogeneity sweep, shapes vs uniform vs oracle, exact finite-horizon regret (spec section 8).
Reg_i = h/2 [(x0^2-V)(1-rho^N)/(1-rho) + N V], rho=(1-theta)^2, V = theta sigma^2/(h^2 (2-theta)), theta_i = mu g_i alpha_i, alpha_i = eta h_i/sqrt(v_i).
Regimes: (a) x0=1, v=1 (psi ~ 1/sigma^2); (b) x0=1, v=sigma^2 (psi const); (c) x0=exp(.5 z''), v=h^2 x0^2+sigma^2.
Strategies: uniform* , Evie* (s_E(g0 s^2)), Amplify* (m_0(g0 s^2), i.e. m_A/m_A(0)), misaligned* ((1+g0 s^2)^(1/2)), oracle, untuned AdamW/Evie/Amplify.
Search: 48-point log grid pre-scan then 80 golden-section iterations; theta_i <= 1.99.
PASS (all gating, ς_h=1 for the sign criteria; ordering and homogeneity on every draw):
 (1) oracle <= every strategy (rel 1e-9) and uniform* <= dense 2000-point grid minimum (rel 1e-9);
 (2) at varsigma=0 all tuned-shape gaps |gap|<1e-6;  (3) regime a: Evie gap >= -1e-3 for >=99 % of draws with varsigma>0, Spearman(varsigma, mean gap)>=0.8;
 (4) regime b: Evie gap < 0 for >=99 % of draws with varsigma>0, Spearman <= -0.8;  (5) misaligned in (a): gap<0 for >=99 %;
 (6) regime c: reported only, plus agreement (100 %) between the surrogate criterion (sum A/g)(sum B g) < (sum A)(sum B) and the surrogate-regret ordering;
 (7) two-sided C2 bound at every coordinate with theta<=1 (rel 1e-9);  (8) power gates g=psi^omega: better iff omega in (0,1), tied at 0 and 1, worse outside."""
from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr

from .common import claim, golden_min, m_0, main_for, s_E

W = 0.85
THMAX = 1.99


def reg_coord(th, h, x0, sig, N):
    th = np.asarray(th, float)
    with np.errstate(all="ignore"):
        rho_log = 2.0 * np.log(np.abs(1.0 - th))
        geo = -np.expm1(N * rho_log) / (th * (2.0 - th))
        V = th * sig ** 2 / (h ** 2 * (2.0 - th))
        reg = 0.5 * h * ((x0 ** 2 - V) * geo + N * V)
    return np.where((th > 0) & (th < 2.0), reg, np.inf)


def tune_scale(shape_alpha, h, x0, sig, N, ngrid=48, iters=80):
    """argmin over mu of sum_i Reg_i(mu * shape_alpha_i), theta<=1.99.  Returns (mu, value)."""
    mu_max = THMAX / shape_alpha.max()
    lg = np.linspace(np.log(mu_max) - 18.0, np.log(mu_max), ngrid)
    vals = np.array([reg_coord(np.exp(l) * shape_alpha, h, x0, sig, N).sum() for l in lg])
    j = int(np.argmin(vals))
    lo, hi = lg[max(j - 1, 0)], lg[min(j + 1, ngrid - 1)]
    f = lambda l: np.array(reg_coord(np.exp(float(l)) * shape_alpha, h, x0, sig, N).sum())
    l_best = float(golden_min(f, np.array(lo), np.array(hi), iters))
    cand = [(f(l_best), l_best), (vals[j], lg[j])]
    v, l = min(cand, key=lambda t: float(t[0]))
    return float(np.exp(l)), float(v)


def oracle(alpha, h, x0, sig, N, ngrid=60, iters=80):
    d = alpha.size
    lg = np.linspace(np.log(1e-9), np.log(THMAX), ngrid)
    th_grid = np.exp(lg)[:, None]
    vals = reg_coord(th_grid, h[None], x0[None], sig[None], N)
    j = np.argmin(vals, axis=0)
    lo, hi = lg[np.maximum(j - 1, 0)], lg[np.minimum(j + 1, ngrid - 1)]
    f = lambda l: reg_coord(np.exp(l), h, x0, sig, N)
    l_best = golden_min(f, lo, hi, iters)
    th = np.exp(l_best)
    th = np.where(f(l_best) <= vals[j, np.arange(d)], th, np.exp(lg[j]))
    return th, reg_coord(th, h, x0, sig, N)


def draw(rng, regime, d, vs, vh):
    z, zp, zpp = rng.standard_normal(d), rng.standard_normal(d), rng.standard_normal(d)
    h = np.exp(vh * z)
    sig = np.exp(vs * zp)
    x0 = np.exp(0.5 * zpp) if regime == "c" else np.ones(d)
    v = np.ones(d) if regime == "a" else (sig ** 2 if regime == "b" else h ** 2 * x0 ** 2 + sig ** 2)
    return h, sig, x0, v


def one_draw(rng, regime, d, vs, vh, N):
    h, sig, x0, v = draw(rng, regime, d, vs, vh)
    a0 = h / np.sqrt(v)
    eta = 0.01 / np.median(a0)
    alpha = eta * a0
    g0 = (1 / W - 1) / np.mean(sig ** 2)
    x = g0 * sig ** 2
    shapes = {"uniform": np.ones(d), "Evie": s_E(x), "Amplify": m_0(x), "misaligned": (1.0 + x) ** 0.5}
    res, thetas = {}, {}
    for name, sh in shapes.items():
        mu, val = tune_scale(sh * alpha, h, x0, sig, N)
        res[name] = val
        thetas[name] = mu * sh * alpha
    th_or, r_or = oracle(alpha, h, x0, sig, N)
    res["oracle"], thetas["oracle"] = float(r_or.sum()), th_or
    for name, th in (("AdamW(m=1)", alpha), ("Evie(mu=1)", s_E(x) * alpha), ("Amplify(mu=1)", m_0(x) / np.sqrt(W) * alpha)):
        res[name] = float(reg_coord(th, h, x0, sig, N).sum())
        thetas[name] = th
    # dense-grid check of uniform*
    mu_max = THMAX / alpha.max()
    dense = min(float(reg_coord(np.exp(l) * alpha, h, x0, sig, N).sum()) for l in np.linspace(np.log(mu_max) - 18, np.log(mu_max), 400))
    # C2 bound at every coordinate with theta<=1
    viol = 0.0
    for name, th in thetas.items():
        ok = (th > 0) & (th <= 1.0)
        if not ok.any():
            continue
        t_, hh, xx, ss = th[ok], h[ok], x0[ok], sig[ok]
        M = hh * xx ** 2 / (4 * t_) + N * t_ * ss ** 2 / (4 * hh)
        R_ = reg_coord(t_, hh, xx, ss, N)
        lo = (1 - 2 * np.exp(-2 * N * t_) - 2 / (N * t_)) * M
        hi = (1 + t_) * M
        viol = max(viol, float(np.max(np.maximum((lo - R_) / M, 0.0))), float(np.max(np.maximum((R_ - hi) / M, 0.0))))
    # surrogate
    A = h * x0 ** 2 / (4 * alpha)
    B = N * alpha * sig ** 2 / (4 * h)
    sur = {}
    def sur_min(gsh):                      # numerically tuned scale on the surrogate M = A/m + B m (independent of the closed form)
        f = lambda l: np.array(((A / (np.exp(float(l)) * gsh)) + B * np.exp(float(l)) * gsh).sum())
        l = golden_min(f, np.array(-40.0), np.array(40.0), 120)
        return float(f(l))
    base = sur_min(np.ones(d))
    for name in ("Evie", "Amplify", "misaligned"):
        gsh = shapes[name]
        crit = (A / gsh).sum() * (B * gsh).sum() < A.sum() * B.sum() * (1 - 1e-9)
        order = sur_min(gsh) < base * (1 - 1e-9)
        tie = abs((A / gsh).sum() * (B * gsh).sum() / (A.sum() * B.sum()) - 1) <= 1e-9
        sur[name] = bool(tie or crit == order)
    return res, dense, viol, sur, (A, B, alpha, h, x0, sig)


def run(rng, smoke, out):
    N = 5000
    ds = [100, 1000] if not smoke else [100]
    stds = [0, 0.25, 0.5, 1.0, 1.5, 2.0]
    draws = 50 if not smoke else 3
    rows = []
    store = {}
    order_viol = uni_viol = bound_viol = 0.0
    homog_max = 0.0
    sur_total = sur_agree = 0
    for vh in (1.0, 0.0):
        for regime in "abc":
            for d in ds:
                for vs in stds:
                    for k in range(draws):
                        res, dense, viol, sur, abd = one_draw(rng, regime, d, vs, vh, N)
                        ru = res["uniform"]
                        for name, val in res.items():
                            rows.append(f"{regime},{d},{vs},{vh},{k},{name},{val},{val / ru}")
                            if name != "oracle":
                                order_viol = max(order_viol, max((res["oracle"] - val) / val, 0.0))
                        uni_viol = max(uni_viol, (ru - dense) / dense)
                        bound_viol = max(bound_viol, viol)
                        gaps = {n: 1 - res[n] / ru for n in ("Evie", "Amplify", "misaligned")}
                        store.setdefault((vh, regime, d, vs), []).append(gaps)
                        if vs == 0:
                            homog_max = max(homog_max, max(abs(g) for g in gaps.values()))
                        if regime == "c" and vh == 1.0:
                            for name, okk in sur.items():
                                sur_total += 1
                                sur_agree += int(okk)
    (out / "table.csv").write_text("regime,d,varsigma,varsigma_h,draw,strategy,regret,ratio_to_uniform\n" + "\n".join(rows) + "\n")
    summ = {}
    figs = {}
    ok_3 = ok_4 = ok_5 = True
    for regime in "abc":
        for d in ds:
            gm = []
            for vs in stds:
                g = store[(1.0, regime, d, vs)]
                gm.append(float(np.mean([x["Evie"] for x in g])))
                if vs > 0:
                    ev = np.array([x["Evie"] for x in g]); mi = np.array([x["misaligned"] for x in g])
                    summ[f"{regime}_d{d}_vs{vs}"] = {"mean_gap_evie": float(ev.mean()), "frac_evie_better": float((ev > 0).mean()),
                                                      "frac_evie_ge_-1e-3": float((ev >= -1e-3).mean()), "frac_evie_neg": float((ev < 0).mean()),
                                                      "frac_mis_neg": float((mi < 0).mean())}
            figs[f"gap_{regime}_d{d}"] = np.array(gm)
            sp = float(spearmanr(stds, gm).statistic) if np.ptp(gm) > 0 else 0.0
            summ[f"{regime}_d{d}_spearman"] = sp
            pos = [s for s in stds if s > 0]
            if regime == "a":
                ok_3 &= all(summ[f"a_d{d}_vs{s}"]["frac_evie_ge_-1e-3"] >= 0.99 for s in pos) and sp >= 0.8
                ok_5 &= all(summ[f"a_d{d}_vs{s}"]["frac_mis_neg"] >= 0.99 for s in pos)
            if regime == "b":
                ok_4 &= all(summ[f"b_d{d}_vs{s}"]["frac_evie_neg"] >= 0.99 for s in pos) and sp <= -0.8
    # power gates (regime c, varsigma=1)
    pg_ok, pg_obs = True, []
    for k in range(draws):
        h, sig, x0, v = draw(rng, "c", ds[0], 1.0, 1.0)
        al = (0.01 / np.median(h / np.sqrt(v))) * h / np.sqrt(v)
        A = h * x0 ** 2 / (4 * al); B = N * al * sig ** 2 / (4 * h)
        psi = A / B
        for om in (-0.5, 0.0, 0.25, 0.5, 0.75, 1.0, 1.5):
            gsh = psi ** om
            rel = ((A / gsh).sum() * (B * gsh).sum()) / (A.sum() * B.sum()) - 1.0
            if om in (0.0, 1.0):
                ok = abs(rel) < 1e-9
            elif 0 < om < 1:
                ok = rel < -1e-12
            else:
                ok = rel > 1e-12
            pg_ok &= bool(ok)
            if k == 0:
                pg_obs.append({"omega": om, "rel_change_of_bound_product": float(rel)})
    c_frac = {f"d{d}_vs{s}": summ.get(f"c_d{d}_vs{s}", {}).get("frac_evie_better") for d in ds for s in stds if s > 0}
    np.savez(out / "fig_data.npz", varsigma=np.array(stds), **figs)
    cl = [
        claim("C2-ordering", "the per-coordinate oracle is never worse than any strategy; uniform* attains the 1-D minimum",
              "max rel violation <= 1e-9", {"oracle_violation": order_viol, "uniform_vs_dense_grid_violation": uni_viol}, order_viol <= 1e-9 and uni_viol <= 1e-9),
        claim("C2-homogeneity", "with equal noise (varsigma=0) a shape cannot beat the uniform scale", "|gap| < 1e-6 for every tuned shape", {"max_abs_gap": homog_max}, homog_max < 1e-6),
        claim("C2-regime-a", "when psi ~ 1/sigma^2 the Evie shape beats the best uniform scale; the advantage grows with the noise spread",
              "gap >= -1e-3 for >=99 % of draws, Spearman(varsigma, mean gap) >= 0.8", {"ok": bool(ok_3), "detail": {k: v for k, v in summ.items() if k.startswith("a_")}}, ok_3),
        claim("C2-converse", "when psi is constant no shape beats uniform: Evie is strictly worse, increasingly with the spread",
              "gap < 0 for >=99 % of draws with varsigma>0, Spearman <= -0.8", {"ok": bool(ok_4), "detail": {k: v for k, v in summ.items() if k.startswith("b_")}}, ok_4),
        claim("C2-misaligned", "a gate with the opposite direction loses in regime a", "gap < 0 for >=99 % of draws", {"ok": bool(ok_5)}, ok_5),
        claim("C2-regime-c", "regime c (descriptive): fraction of draws where Evie beats uniform", "reported only; surrogate criterion consistent with surrogate regret",
              {"frac_evie_better": c_frac, "surrogate_checks": sur_total}, sur_agree == sur_total),
        claim("C2-bound", "Reg is within [(1-2e^{-2N th}-2/(N th))M, (1+th)M] at every coordinate with th<=1", "rel violation <= 1e-9", {"max_violation": bound_viol}, bound_viol <= 1e-9),
        claim("C2-powergates", "power gates g=psi^omega beat uniform iff omega in (0,1), tie at 0 and 1", "sign pattern on every draw", {"first_draw": pg_obs, "draws": draws}, pg_ok),
    ]
    return {"N": N, "d": ds, "varsigma": stds, "draws": draws, "w": W}, cl, ["table.csv", "fig_data.npz"]


if __name__ == "__main__":
    raise SystemExit(main_for("R1d", run))
