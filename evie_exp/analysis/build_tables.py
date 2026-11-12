"""E2 ablation tables from the run JSONs: per-arm means, paired differences, exact Wilcoxon, Holm over the
declared confirmatory family, bootstrap CI, win counts, n_pairs/n_total, diverged counts, HEADLINE-ELIGIBLE banner.

    python -m evie_exp.analysis.build_tables --cell AV --results results --out tables
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from .. import cells as C
from ..protocol.runlog import is_final, load_valid, run_path
from ..protocol.stats import bootstrap_ci, holm, paired_wilcoxon_exact, win_count

FAMILY = [("Evie", "AdamW"), ("FixedGamma", "Evie"), ("EvieAmplify", "AdamW"), ("EvieAmplify", "M1"), ("Evie", "M2"), ("FixedGammaOwn", "Evie")]
ARMS = ["AdamW", "Evie", "FixedGamma", "EvieAmplify", "M1", "M2", "FixedGammaOwn"]


def arm_list(profile: dict) -> list:
    return list(profile["arms"]) + list(profile.get("derived", {})) if "derived" in profile else ARMS


def family_of(profile: dict) -> list:
    return [tuple(x) for x in profile["family"]] if "family" in profile else FAMILY


def lr_map(results: Path, cell: C.Cell, profile: dict, sel: dict) -> dict:
    if "derived" in profile:
        lr = {a: sel[a]["lr"] for a in profile["arms"] if a in sel}
        lr.update({a: sel[src]["lr"] for a, src in profile["derived"].items() if src in sel})
        return lr
    ctl = json.load(open(results / cell.name / "controls.json")) if (results / cell.name / "controls.json").exists() else {}
    lr_of = {a: sel.get(a, {}).get("lr") for a in ("AdamW", "Evie", "FixedGamma", "EvieAmplify")}
    lr_of["M1"], lr_of["M2"], lr_of["FixedGammaOwn"] = ctl.get("lr_M1"), ctl.get("lr_M2"), lr_of["Evie"]
    return lr_of


def collect(results: Path, cell: C.Cell, profile: dict, units: list, seeds: list, sel: dict) -> dict:
    """{arm: {unit: [per-seed primary or nan]}} + status counts."""
    out, counts = {}, {}
    lr_of = lr_map(results, cell, profile, sel)
    for arm in arm_list(profile):
        lr = lr_of.get(arm)
        if lr is None:
            continue
        for u in units:
            vals = []
            for s in seeds:
                r = load_valid(run_path(results, cell.name, "eval", arm, u, s, lr))
                st = r["status"] if r and is_final(r) else "missing"
                counts.setdefault(arm, {}).setdefault(st, 0)
                counts[arm][st] += 1
                vals.append(r["metrics"]["primary"] if st == "ok" else float("nan"))
            out.setdefault(arm, {})[u] = vals
    return out, counts


def unit_means(d: dict, units: list) -> np.ndarray:
    return np.array([np.nanmean(d[u]) if np.any(np.isfinite(d[u])) else float("nan") for u in units], float)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", required=True, choices=list(C.CELLS))
    ap.add_argument("--profile", default="ea")
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="tables")
    ap.add_argument("--fake-data", action="store_true", help="units of a --smoke run")
    ap.add_argument("--max-units", type=int, default=None)
    a = ap.parse_args(argv)
    cell, prof, res = C.CELLS[a.cell], C.load_profile(a.profile), Path(a.results)
    units, seeds = C.eval_units(cell, a.max_units, a.fake_data), C.eval_seeds(cell, prof)
    if a.fake_data:
        seeds = seeds[:1]
    sel = json.load(open(res / cell.name / "lr_selected.json"))
    data, counts = collect(res, cell, prof, units, seeds, sel)
    hib = cell.higher_is_better
    fam = [(x, y) for x, y in family_of(prof) if x in data and y in data]
    pv, rows = {}, {}
    for x, y in fam:
        mx, my = unit_means(data[x], units), unit_means(data[y], units)
        ok = np.isfinite(mx) & np.isfinite(my)
        d = (mx - my)[ok]
        t = paired_wilcoxon_exact(mx[ok], my[ok]) if ok.sum() >= 3 else {"p": float("nan"), "method": "n<3"}
        lo, hi = bootstrap_ci(d)
        rows[(x, y)] = dict(n_pairs=int(ok.sum()), n_total=len(units), mean_diff=float(d.mean()) if len(d) else float("nan"), ci_lo=lo, ci_hi=hi,
                            wins=win_count(mx[ok], my[ok], hib), p=t["p"], method=t["method"],
                            rel_pct=float(np.mean(d) / abs(np.mean(my[ok])) * 100) if len(d) else float("nan"))
        pv[f"{x} vs {y}"] = t["p"]
    ph = holm(pv)
    amp_ok = all(ph.get(k, 1.0) < 0.05 for k in ("EvieAmplify vs AdamW", "EvieAmplify vs M1"))
    lrmap = lr_map(res, cell, prof, sel)
    md = [f"# Ablation {cell.name} ({cell.domain}), primary metric {cell.primary} ({'higher' if hib else 'lower'} is better), profile {a.profile}", "",
          f"HEADLINE-ELIGIBLE: {'yes' if amp_ok else 'no'}  (yes only if EvieAmplify-vs-AdamW AND EvieAmplify-vs-M1 both survive Holm at 0.05)", "",
          "## Arms (mean ± SD over paired units of the per-unit seed-mean)", "", "| arm | mean | sd | status counts (runs) | LR |", "|---|---|---|---|---|"]
    lrs = {**{k: v["lr"] for k, v in sel.items()}}
    for arm in arm_list(prof):
        if arm in data:
            m = unit_means(data[arm], units)
            md.append(f"| {arm} | {np.nanmean(m):.5g} | {np.nanstd(m):.3g} | {counts.get(arm, {})} | {lrmap.get(arm, 'derived')}{'' if arm in lrs else ' (derived)'} |")
    md += ["", "## Confirmatory family (Holm over all rows below; the rest is descriptive)", "",
           "| comparison | mean diff (first - second) | 95% CI | wins | n_pairs/n_total | raw p | Holm p | significant |", "|---|---|---|---|---|---|---|---|"]
    for (x, y), r in rows.items():
        k = f"{x} vs {y}"
        sig = "yes" if ph.get(k, 1.0) < 0.05 else "no"
        md.append(f"| {k} | {r['mean_diff']:+.5g} | [{r['ci_lo']:+.4g}, {r['ci_hi']:+.4g}] | {r['wins']} | {r['n_pairs']}/{r['n_total']} | {r['p']:.4g} | {ph.get(k, float('nan')):.4g} | {sig} |")
    md += ["", "Relative % differences (descriptive only): " + "; ".join(f"{x} vs {y}: {r['rel_pct']:+.2f}%" for (x, y), r in rows.items())]
    md += ["", "## Diagnostics of the Evie-family arms (mean over runs of the last evaluation interval)", "",
           "| arm | rho_rms | amplify_frac | floor_frac | cap_frac | expo>1 frac | gamma0 (mean) | calibration clip status |", "|---|---|---|---|---|---|---|---|"]
    for arm in arm_list(prof):
        if arm not in data or arm == "AdamW":
            continue
        lrv = lrmap.get(arm)
        recs = [load_valid(run_path(res, cell.name, "eval", arm, u, sd, lrv)) for u in units for sd in seeds]
        recs = [r for r in recs if r and r["status"] == "ok"]
        if not recs:
            continue
        last = lambda k: np.nanmean([r["diagnostics"]["timeseries"][k][-1] for r in recs])
        g0 = [r["diagnostics"]["calibration"]["gamma0"] for r in recs if r["diagnostics"]["calibration"]["gamma0"] is not None]
        stc = {}
        for r in recs:
            c = r["diagnostics"]["calibration"]["clip_status"]; stc[c] = stc.get(c, 0) + 1
        md.append(f"| {arm} | {last('rho_rms'):.4f} | {last('amplify_frac'):.3f} | {last('floor_frac'):.2g} | {last('cap_frac'):.2g} | {last('expo_gt1_frac'):.3g} | {np.mean(g0) if g0 else float('nan'):.4g} | {stc} |")
    md += ["", "(floor_frac and cap_frac must be 0 for the Evie-Amplify arms: the clamps are inactive by theorem B2.)"]
    edge = [k for k, v in sel.items() if v.get("edge_warning")]
    md += ["", f"LR edge warnings: {edge or 'none'}"]
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    (out / f"{cell.name}_summary.md").write_text("\n".join(md) + "\n")
    json.dump({"holm": ph, "rows": {f"{x} vs {y}": r for (x, y), r in rows.items()}, "headline_eligible": amp_ok, "counts": counts},
              open(out / f"{cell.name}_summary.json", "w"), indent=1, default=float)
    print("\n".join(md))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
