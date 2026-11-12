"""R1b (A1, A2, A4, A5, B5): control-model simulations.  NOTE (header, required by the spec): plain long-horizon Monte Carlo of
E exp(gamma C_T) has exponentially growing variance in T, so the cost-rate minimiser (ii) uses the EXACT discrete Bellman recursion
(exact for the Euler-Maruyama system, B5); Monte Carlo is used only where it is trustworthy (short horizon).

(i)   A5   stationary variance: |estimate - exact Euler q/(k(2-k dt))| <= 4 SE in every setting; continuous q/(2k*) within 1.5 %;
           strictly increasing in kappa; risk-neutral cost rate (q/4)(H/k+Rk) >= (q/2)sqrt(HR), equality only at kappa q = 0.
(ii)  A4/B5 lambda_eta(k) -> lambda(k) (max rel err at the smallest eta < 1 % is REPORTED but does not gate - see Q12; gates: log-log slope in [0.9,1.15], successive error ratios in [1.5, 2.6] for the eta
           ratios 2 (the 2.5 ratio is checked against [1.5, 3.2])); argmin_k lambda_eta within one grid spacing of the exact discrete
           optimum k_d(eta) and |k_d-(k*-eta H/(2R))|/k* < 5 eta^2; lambda* = q R k*/2 (rel 1e-10); lambda'' = delta q H/(2 D^3) (rel 1e-8);
           MC of the finite-horizon recursion within 4 SE.
           k-range rule (fixed in advance): k in [max(0.3k*, 1.15 k_dom), 3k*] with k_dom = sqrt(gamma q H/delta) the domain edge.
(iii) A2   |J_disc(u*)-J*| <= 2 dt and halves with dt; J_disc(u_eps) > J_disc(u*); MC differences > 4 SE for |eps| >= 0.2
           (common random numbers); MC J(u*) within 4 SE of J_disc(u*).
(iv)  A1   J_gamma monotone in gamma, J_{gamma<0} <= E C <= J_{gamma>0}; err(0.1)/err(0.01) in [50, 200] (else recorded only)."""
from __future__ import annotations

import numpy as np

from .common import claim, main_for
from .r1a_riccati import closed_S, closed_c


def kstar(H, R, delta):
    return np.sqrt(H / (R * delta))


def lam_cont(k, g, q, H, R):
    c = H + R * k ** 2
    if g == 0:
        return q * c / (4 * k)
    return (k - np.sqrt(k ** 2 - g * q * c)) / (2 * g)


def lam_eta(k, g, q, H, R, eta):
    """Exact Euler-Maruyama cost rate via the fixed point of P <- eta c + a^2 P/(1-g eta q P) (B5); closed-form root, iteration-checked."""
    c = H + R * k ** 2
    a2 = (1 - eta * k) ** 2
    if g == 0:
        P = eta * c / (1 - a2)
        return q * P / 2, P, np.ones_like(k, bool)
    B = 1 - a2 + g * eta ** 2 * q * c
    disc = B ** 2 - 4 * g * eta ** 2 * q * c
    ok = (disc >= 0) & (B > 0)
    disc = np.where(ok, disc, 0.0)
    P = (B - np.sqrt(disc)) / (2 * g * eta * q)
    ok &= (1 - g * eta * q * P > 0)
    arg = np.where(ok, 1 - g * eta * q * P, 1.0)
    return -np.log(arg) / (2 * g * eta), P, ok


def disc_recursion_iter(k, g, q, H, R, eta, n=200000):
    c = H + R * k ** 2
    P = np.zeros_like(k)
    for _ in range(n):
        den = 1 - g * eta * q * P
        if np.any(den <= 0):
            return None
        Pn = eta * c + (1 - eta * k) ** 2 * P / den
        if np.max(np.abs(Pn - P)) < 1e-15 * max(1.0, np.max(np.abs(Pn))):
            return Pn
        P = Pn
    return P


def finite_horizon_J(kgain, g, q, H, R, dt, x0, nsteps):
    """Time-varying gain k_n (array length nsteps, step n = time n dt): backward recursion, J = P0 x0^2/2 + d0."""
    P, d = 0.0, 0.0
    for n in range(nsteps - 1, -1, -1):
        k = kgain[n]
        c = H + R * k ** 2
        den = 1 - g * dt * q * P
        if den <= 0:
            return np.inf
        d = d - (np.log(den) / (2 * g) if g != 0 else -0.5 * dt * q * P)
        P = dt * c + (1 - dt * k) ** 2 * P / den
    return 0.5 * P * x0 ** 2 + d


def simulate_costs(rng, kfun_list, q, H, R, dt, nsteps, N, x0, chunk=50000):
    """Euler-Maruyama paths with common random numbers; kfun_list[j](n) -> gain of policy j at step n. Returns C (P, N)."""
    P = len(kfun_list)
    C = np.empty((P, N))
    done = 0
    while done < N:
        m = min(chunk, N - done)
        x = np.full((P, m), x0, float)
        acc = np.zeros((P, m))
        for n in range(nsteps):
            kv = np.array([f(n) for f in kfun_list])[:, None]
            acc += 0.5 * (H * x ** 2 + R * (kv * x) ** 2) * dt
            z = rng.standard_normal(m)[None, :]
            x = (1 - kv * dt) * x + np.sqrt(q * dt) * z
        C[:, done:done + m] = acc
        done += m
    return C


def Jmc(C, g):
    if g == 0:
        return C.mean(axis=-1), C.std(axis=-1, ddof=1) / np.sqrt(C.shape[-1]), (C - C.mean(axis=-1, keepdims=True))
    mx = (g * C).max(axis=-1, keepdims=True)
    w = np.exp(g * C - mx)
    m = w.mean(axis=-1, keepdims=True)
    J = (np.log(m[..., 0]) + mx[..., 0]) / g
    infl = (w / m - 1.0) / g
    return J, infl.std(axis=-1, ddof=1) / np.sqrt(C.shape[-1]), infl


def run(rng, smoke, out):
    claims, files = [], []
    H = R = 1.0
    # ---------------- (i) A5 -------------------------------------------------------------------------------------
    M, nt = (8192, 100) if not smoke else (512, 20)
    settings = []
    for q in (0.5, 2.0):
        for kap in (0.0, 0.5, 2.0, 8.0):
            settings.append((q, -kap, 1 + kap * q, f"kappa={kap}"))
        for gq in (0.25, 0.5):
            settings.append((q, gq / q, 1 - gq, f"averse g*q={gq}"))
    rows, all_ok, cont_ok, worst_z, worst_cont = [], True, True, 0.0, 0.0
    exact_by_q = {}
    for q, g, delta, lab in settings:
        ks = float(kstar(H, R, delta))
        dt = 0.01 / ks
        T = 40 / ks
        nsteps = int(round(T / dt))
        x = np.zeros(M)
        recs = []
        for n in range(nsteps):
            x = (1 - ks * dt) * x + np.sqrt(q * dt) * rng.standard_normal(M)
            if n >= nsteps // 2 and (n - nsteps // 2) % max(1, (nsteps - nsteps // 2) // nt) == 0:
                recs.append(x.reshape(16, -1).var(axis=1, ddof=1))
        grp = np.mean(recs, axis=0)
        est, se = float(grp.mean()), float(grp.std(ddof=1) / 4)
        v_ex = q / (ks * (2 - ks * dt))
        v_cont = q / (2 * ks)
        z = abs(est - v_ex) / se
        worst_z = max(worst_z, z)
        rel_c = abs(est - v_cont) / v_cont
        worst_cont = max(worst_cont, rel_c)
        all_ok &= z <= 4.0
        cont_ok &= rel_c < 0.015
        rate_est = 0.5 * (H + R * ks ** 2) * est
        rate_formula = q / 4 * (H / ks + R * ks)
        rows.append(f"{q},{g},{lab},{ks},{est},{se},{v_ex},{v_cont},{rate_est},{rate_formula}")
        if g <= 0:
            exact_by_q.setdefault(q, []).append((-g, v_cont, rate_formula))
    inc = all(all(np.diff([v for _, v, _ in sorted(v)]) > 0) for v in exact_by_q.values())
    rn_ok = True
    for q, v in exact_by_q.items():
        for kap, _, rf in v:
            lo = q / 2 * np.sqrt(H * R)
            rn_ok &= (rf > lo * (1 + 1e-12)) if kap > 0 else abs(rf - lo) < 1e-12
    (out / "table_A5.csv").write_text("q,gamma,label,k_star,var_est,SE,var_exact_euler,var_cont,cost_rate_est,cost_rate_formula\n" + "\n".join(rows) + "\n")
    files.append("table_A5.csv")
    claims.append(claim("A5", "stationary variance of the closed loop: exact Euler value within 4 SE; continuous q/(2k*) within 1.5 %; increasing in kappa; "
                        "risk-neutral cost rate >= (q/2) sqrt(HR), equality only at kappa=0",
                        "|est-exact|<=4SE all settings; continuous<1.5 %; strictly increasing; inequality strict for kappa>0",
                        {"max_z": worst_z, "max_rel_dev_continuous": worst_cont, "increasing_in_kappa": bool(inc), "risk_neutral_inequality": bool(rn_ok),
                         "paths": M, "settings": len(settings)}, all_ok and cont_ok and inc and rn_ok))
    # ---------------- (ii) A4/B5 ---------------------------------------------------------------------------------
    q = 1.0
    gammas = [-2.0, -0.5, 0.0, 0.25, 0.5]
    etas = np.array([0.1, 0.05, 0.02, 0.01, 0.005])
    chk1_ok = chk2_ok = chk3_ok = literal_1pct = True
    obs1, obs2, obs3, tab = [], [], [], []
    for g in gammas:
        delta = 1 - g * R * q
        ks = float(kstar(H, R, delta))
        kdom = np.sqrt(max(g, 0.0) * q * H / delta)
        klo = max(0.3 * ks, 1.15 * kdom)
        kg = np.exp(np.linspace(np.log(klo), np.log(3 * ks), 201))
        spacing = np.log(kg[1] / kg[0])
        lc = lam_cont(kg, g, q, H, R)
        errs = []
        for eta in etas:
            le, P, ok = lam_eta(kg, g, q, H, R, eta)
            feas = bool(np.all(ok))
            e = float(np.max(np.abs(le - lc) / lc)) if feas else np.inf
            errs.append(e)
            # exact discrete optimum by bounded scalar minimisation, then compare
            from scipy.optimize import minimize_scalar
            f = lambda lk: float(lam_eta(np.array([np.exp(lk)]), g, q, H, R, eta)[0][0]) if lam_eta(np.array([np.exp(lk)]), g, q, H, R, eta)[2][0] else 1e30
            r = minimize_scalar(f, bounds=(np.log(klo), np.log(3 * ks)), method="bounded", options={"xatol": 1e-12})
            kd = float(np.exp(r.x))
            k_grid_min = float(kg[int(np.argmin(np.where(ok, le, np.inf)))])
            within = abs(np.log(k_grid_min / kd)) <= spacing * 1.0001
            resid = abs(kd - (ks - eta * H / (2 * R))) / ks
            chk2_ok &= bool(within) and resid < 5 * eta ** 2
            tab.append(f"{g},{eta},{e},{kd},{k_grid_min},{resid},{resid / eta ** 2}")
            if eta == etas[0] and g > 0:
                pass
        # iteration cross-check of the closed-form root at eta=0.1
        Pit = disc_recursion_iter(kg[::20], g, q, H, R, 0.1)
        Pcl = lam_eta(kg[::20], g, q, H, R, 0.1)[1]
        iter_err = float(np.max(np.abs(Pit - Pcl) / Pcl)) if Pit is not None else np.inf
        ratios = [errs[i] / errs[i + 1] for i in range(len(errs) - 1)]
        er_ok = all((1.5 <= r_ <= 2.6) if abs(etas[i] / etas[i + 1] - 2) < 1e-9 else (1.5 <= r_ <= 3.2) for i, r_ in enumerate(ratios))
        slope = float(np.polyfit(np.log(etas), np.log(errs), 1)[0])
        chk1_ok &= er_ok and iter_err < 1e-9 and 0.9 <= slope <= 1.15
        literal_1pct &= errs[-1] < 0.01
        obs1.append({"gamma": g, "err_smallest_eta": errs[-1], "k_max_over_kstar": 3.0, "err_ratios": ratios, "log_slope": slope, "iter_vs_closed_form": iter_err})
        lam_star = lam_cont(np.array([ks]), g, q, H, R)[0]
        e_star = abs(lam_star - 0.5 * q * R * ks) / (0.5 * q * R * ks)
        # lambda'' by 5-point stencil vs delta q H/(2 D^3)
        h = 2e-3 * ks
        fk = lambda k_: lam_cont(np.array([k_]), g, q, H, R)[0]
        d2 = (-fk(ks + 2 * h) + 16 * fk(ks + h) - 30 * fk(ks) + 16 * fk(ks - h) - fk(ks - 2 * h)) / (12 * h ** 2)
        D = np.sqrt(ks ** 2 * delta - g * q * H)
        d2_th = delta * q * H / (2 * D ** 3)
        e2 = abs(d2 - d2_th) / d2_th
        chk3_ok &= e_star < 1e-10 and e2 < 1e-8 and d2_th > 0
        obs3.append({"gamma": g, "lambda_star_rel_err": float(e_star), "lambda_pp_rel_err": float(e2)})
    (out / "table_B5.csv").write_text("gamma,eta,max_rel_err_lambda,k_d,k_grid_argmin,resid_over_kstar,resid_over_eta2\n" + "\n".join(tab) + "\n")
    files.append("table_B5.csv")
    claims.append(claim("A4", "lambda(k) closed form; minimiser k*; lambda* = q R k*/2; lambda'' = delta q H/(2 D^3) > 0",
                        "lambda* rel err<1e-10; lambda'' rel err<1e-8", {"per_gamma": obs3}, chk3_ok))
    claims.append(claim("B5", "exact Euler cost rate lambda_eta converges to lambda linearly in eta; discrete optimum k_d = k* - eta H/(2R) + O(eta^2)",
                        "log-log slope of the error in [0.9,1.15]; successive error ratios in [1.5,2.6] (eta ratio 2) / [1.5,3.2] (ratio 2.5); closed-form root = iteration (1e-9); |k_d-(k*-etaH/2R)|/k*<5 eta^2; argmin within one grid cell",
                        {"convergence": obs1, "k_d_residual_ok": bool(chk2_ok), "literal_spec_err_below_1pct_at_eta_min": bool(literal_1pct), "note": "the relative error is ~ eta*k/2 and the k grid reaches 3k*, so for k*>1.33 the literal 1 % bound at eta=0.005 cannot hold; reported, not gating (OPEN_QUESTIONS Q12)"}, chk1_ok and chk2_ok))
    # MC validation of the finite-horizon recursion
    Nmc = 400000 if not smoke else 20000
    ks = 1.0
    T, dt = 2 / ks, 0.01 / ks
    ns = int(round(T / dt))
    C = simulate_costs(rng, [lambda n: ks], 1.0, H, R, dt, ns, Nmc, 1.0)[0]
    mc_ok, mc_obs = True, []
    for g in (-1.0, -0.3, 0.2):
        Jd = finite_horizon_J(np.full(ns, ks), g, 1.0, H, R, dt, 1.0, ns)
        J, se, _ = Jmc(C[None, :], g)
        z = abs(J[0] - Jd) / se[0]
        mc_ok &= z <= 4
        mc_obs.append({"gamma": g, "J_disc": Jd, "J_mc": float(J[0]), "SE": float(se[0]), "z": float(z)})
    claims.append(claim("B5-MC", "short-horizon Monte Carlo of the Euler system equals the finite-horizon Bellman recursion", "|J_MC-J_disc|<=4 SE", {"cases": mc_obs}, mc_ok))
    # ---------------- (iii) A2 -----------------------------------------------------------------------------------
    T, x0 = 2.0, 1.0
    gam = [-1.0, -0.3, 0.0, 0.3]
    eps_list = [-0.5, -0.2, 0.2, 0.5]
    qq = 1.0
    def kopt(g, t):
        a = 1 / R - g * qq
        return closed_S(a, H, T - t) / R
    def Jstar(g):
        a = 1 / R - g * qq
        return 0.5 * closed_S(a, H, T) * x0 ** 2 + closed_c(a, H, qq, T)
    obs_a, a_ok = [], True
    for g in gam:
        gaps = []
        for dt_ in (0.004, 0.002, 0.001):
            ns_ = int(round(T / dt_))
            tt = np.arange(ns_) * dt_
            kg_ = np.array([kopt(g, t) for t in tt])
            gaps.append(abs(finite_horizon_J(kg_, g, qq, H, R, dt_, x0, ns_) - Jstar(g)))
        ratio = [gaps[i] / gaps[i + 1] for i in range(2)]
        a_ok &= gaps[1] <= 2 * 0.002 and all(1.5 <= r_ <= 2.6 for r_ in ratio)
        obs_a.append({"gamma": g, "gaps": gaps, "ratios": ratio})
    claims.append(claim("A2-disc", "the discrete cost of the optimal time-varying gain converges to J* = S(T)x0^2/2 + c(T) linearly in dt",
                        "gap(dt=0.002)<=2 dt and successive ratios in [1.5,2.6]", {"per_gamma": obs_a}, a_ok))
    dt = 0.002 if not smoke else 0.01
    ns = int(round(T / dt))
    tt = np.arange(ns) * dt
    Nmc = 500000 if not smoke else 20000
    pols, labels = [], []
    for g in gam:
        base = np.array([kopt(g, t) for t in tt])
        ks_ = float(kstar(H, R, 1 - g * qq))
        pols += [base] + [base * (1 + e) for e in eps_list] + [np.full(ns, ks_)]
        labels += [(g, "opt")] + [(g, f"eps={e}") for e in eps_list] + [(g, "const k*")]
    funs = [(lambda n, arr=arr: arr[n]) for arr in pols]
    C = simulate_costs(rng, funs, qq, H, R, dt, ns, Nmc, x0, chunk=25000)
    per = 2 + len(eps_list)
    pos_ok, ustar_ok, disc_ok, mc_rows = True, True, True, []
    for gi, g in enumerate(gam):
        sl = slice(gi * per, (gi + 1) * per)
        J, se, infl = Jmc(C[sl], g)
        Jd = [finite_horizon_J(pols[gi * per + j], g, qq, H, R, dt, x0, ns) for j in range(per)]
        ustar_ok &= abs(J[0] - Jd[0]) <= 4 * se[0]
        for j in range(1, per):
            disc_ok &= Jd[j] > Jd[0]
            diff = J[j] - J[0]
            dse = float(np.std(infl[j] - infl[0], ddof=1) / np.sqrt(Nmc))
            big = labels[gi * per + j][1] == "const k*" or abs(float(labels[gi * per + j][1].split("=")[1])) >= 0.2
            if big:
                pos_ok &= diff > 4 * dse
            mc_rows.append({"gamma": g, "policy": labels[gi * per + j][1], "J_disc_diff": Jd[j] - Jd[0], "J_mc_diff": float(diff), "SE_crn": dse})
    claims.append(claim("A2", "the Riccati gain minimises J among perturbed and constant gains (exact discrete recursion)", "J_disc(u_eps)>J_disc(u*) for all eps != 0, all gamma",
                        {"all_larger": bool(disc_ok)}, disc_ok))
    claims.append(claim("A2-MC", "Monte Carlo with common random numbers sees the same ordering and the same optimal value",
                        "MC differences > 4 SE for |eps|>=0.2 and constant k*; MC J(u*) within 4 SE of J_disc(u*)",
                        {"mc_J_star_matches": bool(ustar_ok), "ordering_resolved": bool(pos_ok), "rows": mc_rows, "paths": Nmc}, ustar_ok and pos_ok))
    # ---------------- (iv) A1 ------------------------------------------------------------------------------------
    Nmc = 10 ** 6 if not smoke else 40000
    ks = 1.0
    dt = 0.01 / ks if not smoke else 0.05
    ns = int(round(5 / ks / dt))
    C = simulate_costs(rng, [lambda n: ks], 1.0, H, R, dt, ns, Nmc, 1.0)[0]
    gl = [-0.1, -0.03, -0.01, 0.01, 0.03, 0.1]
    Jg, errs = [], {}
    EC, VC = C.mean(), C.var()
    for g in gl:
        J = Jmc(C[None, :], g)[0][0]
        Jg.append(J)
        errs[g] = abs(J - (EC + g / 2 * VC))
    mono = bool(np.all(np.diff(Jg) > 0))
    order = all(J <= EC + 1e-12 for g, J in zip(gl, Jg) if g < 0) and all(J >= EC - 1e-12 for g, J in zip(gl, Jg) if g > 0)
    ratio = errs[0.1] / errs[0.01] if errs[0.01] > 0 else np.nan
    claims.append(claim("A1", "J_gamma = (1/gamma) log E exp(gamma C) is nondecreasing in gamma, J_{<0} <= E C <= J_{>0}, and equals E C + gamma Var C/2 + O(gamma^2)",
                        "monotone and ordered for all pairs (ratio err(0.1)/err(0.01) in [50,200] is recorded; it gates only if finite)",
                        {"J": dict(zip(map(str, gl), map(float, Jg))), "E_C": float(EC), "expansion_err_ratio_0.1_over_0.01": float(ratio), "paths": Nmc},
                        mono and order and (not np.isfinite(ratio) or 50 <= ratio <= 200)))
    return {"paths_A5": M, "etas": etas.tolist(), "gammas": gammas}, claims, files


if __name__ == "__main__":
    raise SystemExit(main_for("R1b", run))
