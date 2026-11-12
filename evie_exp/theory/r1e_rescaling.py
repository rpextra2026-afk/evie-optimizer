"""R1e (B3): loss-rescaling invariance L -> cL, float64.  Task: MLP 10-32-1 (tanh), synthetic regression, 2000 points, batch 64 from a fixed index sequence,
T=400, constant LR 1e-2, wd 0, calibration after the step with loop index 125.  All variants use eps=1e-30 so only the hypothesis under test can break invariance.
Variants: AdamW; EvieGateOff / AmplifyGateOff (warmup gate off, calibrated); EvieShipped / AmplifyShipped (gamma0=1 during warm-up);
EvieFixedGamma0 (gamma0=100, never calibrated); FixedGamma (s_F, constant gamma = Evie's calibrated gamma0 at the reference scale).
Scale sets: narrow c = c_ref*{.5,1,2,4} with sigma_bar_f^2(c_ref)=3.9e-4 (probe); wide c in {.1,1,10,100}. Each set with clipping off and on (norm 5 on the scaled loss).
Metric: max_t max_i |theta^(c)-theta^(ref)| / max|theta^(ref)| and the first step above 1e-8.
PASS: narrow, clip off: AdamW, EvieGateOff, AmplifyGateOff <= 1e-10 at every c; EvieShipped/AmplifyShipped >= 1e-6 at some c (H3 breaks);
EvieFixedGamma0 >= 1e-6 at some c; wide, clip off: for the calibrated gate-off variants, deviation > 1e-8 exactly where the calibration clip status
of gamma0 is not 'inactive' at c or at the reference (H4).  FALSIFIED: a calibrated gate-off variant deviating > 1e-8 inside the window with clipping off.
Reported: whether clipping at every step restores invariance."""
from __future__ import annotations

import numpy as np
import torch

from ..optim.evie_family import EvieFamily
from .common import claim, main_for

T, CAL = 400, 125
VARIANTS = ["AdamW", "EvieGateOff", "AmplifyGateOff", "EvieShipped", "AmplifyShipped", "EvieFixedGamma0", "FixedGamma"]


def make_data(seed=1234):
    g = np.random.Generator(np.random.PCG64(seed))
    X = g.standard_normal((2000, 10))
    y = np.sin(X[:, :3].sum(1)) + 0.5 * X[:, 3] * X[:, 4] + 0.1 * g.standard_normal(2000)
    idx = np.stack([g.integers(0, 2000, 64) for _ in range(T)])
    init = [g.standard_normal((32, 10)) / np.sqrt(10), np.zeros(32), g.standard_normal((1, 32)) / np.sqrt(32), np.zeros(1)]
    return X, y, idx, init


def build_opt(variant, params, fixed_gamma=None):
    kw = dict(lr=1e-2, weight_decay=0.0, eps=1e-30, diag=False)
    if variant == "AdamW":
        return EvieFamily(params, mode="adamw", **kw)
    if variant == "EvieGateOff":
        return EvieFamily(params, mode="evie", warmup_gate="off", **kw)
    if variant == "AmplifyGateOff":
        return EvieFamily(params, mode="amplify", warmup_gate="off", **kw)
    if variant == "EvieShipped":
        return EvieFamily(params, mode="evie", warmup_gate="gamma0", gamma0_warmup=1.0, **kw)
    if variant == "AmplifyShipped":
        return EvieFamily(params, mode="amplify", warmup_gate="gamma0", gamma0_warmup=1.0, **kw)
    if variant == "EvieFixedGamma0":
        return EvieFamily(params, mode="evie", warmup_gate="gamma0", gamma0_warmup=100.0, calibrate=False, **kw)
    if variant == "FixedGamma":
        return EvieFamily(params, mode="fixedgamma", fixed_gamma=fixed_gamma, **kw)
    raise ValueError(variant)


def train(variant, c, clip, data, fixed_gamma=None):
    X, y, idx, init = data
    Xt, yt = torch.tensor(X, dtype=torch.float64), torch.tensor(y, dtype=torch.float64)
    params = [torch.tensor(a, dtype=torch.float64, requires_grad=True) for a in init]
    opt = build_opt(variant, params, fixed_gamma)
    traj, clipped = [], 0
    for t in range(T):
        xb, yb = Xt[idx[t]], yt[idx[t]]
        h = torch.tanh(xb @ params[0].T + params[1])
        pred = (h @ params[2].T + params[3]).squeeze(-1)
        loss = c * ((pred - yb) ** 2).mean()
        for p in params:
            p.grad = None
        loss.backward()
        if clip:
            gn = torch.sqrt(sum((p.grad ** 2).sum() for p in params))
            if gn > 5.0:
                clipped += 1
                for p in params:
                    p.grad.mul_(5.0 / gn)
        opt.step()
        if t == CAL:
            opt.calibrate_gamma0()
        traj.append(torch.cat([p.detach().flatten() for p in params]).numpy().copy())
    return np.stack(traj), opt.calib, clipped / T


def deviation(a, ref):
    d = np.abs(a - ref).max(axis=1) / np.abs(ref).max()
    first = int(np.argmax(d > 1e-8)) if np.any(d > 1e-8) else -1
    return float(d.max()), first


def run(rng, smoke, out):
    data = make_data()
    # probe: sigma_bar_f^2 at c=1 (Adam trajectories are scale invariant, so sigma_bar_f^2(c) = c^2 sigma_bar_f^2(1))
    _, calib1, _ = train("EvieGateOff", 1.0, False, data)
    s1 = calib1["sigma_bar_f2"]
    c_ref = float(np.sqrt(3.9e-4 / s1))
    sets = {"narrow": [c_ref * m for m in (0.5, 1, 2, 4)], "wide": [0.1, 1.0, 10.0, 100.0]}
    ref_idx = {"narrow": 1, "wide": 1}
    variants = VARIANTS if not smoke else ["AdamW", "EvieGateOff", "EvieShipped"]
    fixed_gamma = calib1["gamma0"]
    res, rows = {}, []
    for sname, cs in sets.items():
        for clip in (False, True):
            for v in variants:
                runs = [train(v, c, clip, data, fixed_gamma) for c in cs]
                ref = runs[ref_idx[sname]][0]
                for c, (tr, cal, frac) in zip(cs, runs):
                    dev, first = deviation(tr, ref)
                    res[(sname, clip, v, c)] = (dev, first, cal.get("clip_status"), frac)
                    rows.append(f"{sname},{int(clip)},{v},{c},{dev},{first},{cal.get('clip_status')},{cal.get('gamma0')},{frac}")
    (out / "table.csv").write_text("set,clip,variant,c,max_dev,first_step_above_1e-8,gamma0_clip_status,gamma0,frac_steps_clipped\n" + "\n".join(rows) + "\n")
    nar = lambda v, clip=False: [res[("narrow", clip, v, c)][0] for c in sets["narrow"]]
    inv_ok = all(max(nar(v)) <= 1e-10 for v in ("AdamW", "EvieGateOff", "AmplifyGateOff") if v in variants)
    brk = {v: max(nar(v)) >= 1e-6 for v in ("EvieShipped", "AmplifyShipped", "EvieFixedGamma0") if v in variants}
    falsified = any(max(nar(v)) > 1e-8 for v in ("EvieGateOff", "AmplifyGateOff") if v in variants)
    wide_obs, wide_ok = [], True
    for v in ("EvieGateOff", "AmplifyGateOff"):
        if v not in variants:
            continue
        cs = sets["wide"]
        st_ref = res[("wide", False, v, cs[1])][2]
        for c in cs:
            dev, first, st, _ = res[("wide", False, v, c)]
            expect_break = (st != "inactive") or (st_ref != "inactive")
            broke = dev > 1e-8
            if c != cs[1]:
                wide_ok &= (broke == expect_break)
            wide_obs.append({"variant": v, "c": c, "clip_status": st, "ref_status": st_ref, "max_dev": dev, "broke": bool(broke), "expected_break": bool(expect_break)})
    clip_obs = {f"{v}@{sn}": [res[(sn, True, v, c)][0] for c in sets[sn]] for sn in sets for v in ("EvieGateOff",) if v in variants}
    clip_frac = {sn: [res[(sn, True, "EvieGateOff", c)][3] for c in sets[sn]] for sn in sets if "EvieGateOff" in variants}
    cl = [
        claim("B3", "calibrated gate-off Evie / Evie-Amplify and AdamW (eps=0) are invariant to loss rescaling inside the clip-inactive window (clipping off)",
              "narrow set: deviation <= 1e-10; falsified if > 1e-8",
              {"c_ref": c_ref, "sigma_bar_f2_at_c1": s1, "narrow_devs": {v: nar(v) for v in variants}}, bool(inv_ok) and not falsified),
        claim("B3-H3", "the shipped warm-up (gamma0=1) and a fixed gamma0 break invariance", "deviation >= 1e-6 at some c (narrow, clip off)", {"broken": brk}, all(brk.values())),
        claim("B3-H4", "outside the clip-inactive window the calibrated gate-off variants lose invariance exactly where the calibration clip is active",
              "wide set, clip off: break (>1e-8) iff clip status != inactive at c or at the reference", {"rows": wide_obs}, bool(wide_ok)),
        claim("B3-clip", "reported: effect of gradient clipping (norm 5 on the scaled loss)", "descriptive", {"EvieGateOff_dev_with_clipping": clip_obs, "fraction_steps_clipped": clip_frac}, True),
    ]
    return {"T": T, "calibration_index": CAL, "c_ref": c_ref, "sets": sets}, cl, ["table.csv"]


if __name__ == "__main__":
    raise SystemExit(main_for("R1e", run))
