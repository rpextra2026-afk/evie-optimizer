"""E1 offline comparison (spec 6.4): AdaBelief (new checkpoint) vs the existing columns, per domain.

    python -m evie_exp.analysis.compare_adabelief --domain finance \
        --adabelief finance_run1d_adabelief_checkpoint.json \
        --existing A=run1a_checkpoint.json B=run1b_checkpoint.json C=run1c_checkpoint.json --out tables/

Reads the per-seed values from the checkpoint JSONs (the final JSONs hold seed-medians only).
Everything is computed from stored runs; nothing is typed in.  NaN runs are counted and printed.
vision has n=3 pairs (smallest exact two-sided Wilcoxon p = 0.25): no significance marker is printed there.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from ..protocol.stats import holm, min_exact_p, paired_wilcoxon_exact

DOM = {
    "vision": dict(unit_kind="seed", primary="top1", higher=True, keys=['top1', 'top5', 'valloss', 'iterstotgt', 'wall', 'peakmemmb', 'gflops_st', 'exaflops', 'stepsize', 'clipfreq', 'hessnorm', 'hesscv', 'regret'],
                   lower={'top1': False, 'top5': False, 'valloss': True, 'iterstotgt': True, 'wall': True, 'peakmemmb': True, 'gflops_st': True, 'exaflops': True, 'stepsize': None, 'clipfreq': True, 'hessnorm': None, 'hesscv': True, 'regret': True},
                   opts={"AdamW": "AdamW", "Evie": "Evie", "EvieAmplify": "Evie-Amplify"}),
    "finance": dict(unit_kind="ticker", primary="test_mse", higher=False, keys=['test_mse', 'test_rmse', 'dir_acc', 'steps_half', 'steps_eps', 'wall_s', 'peak_mem_mb', 'gflops', 'exaflops', 'mean_step_sz', 'clip_trigger_freq', 'hess_diag_norm', 'hess_diag_cv', 'regret', 'tvi', 'sce'],
                    lower={'test_mse': True, 'test_rmse': True, 'dir_acc': False, 'steps_half': True, 'steps_eps': True, 'wall_s': True, 'peak_mem_mb': True, 'gflops': True, 'exaflops': True, 'mean_step_sz': None, 'clip_trigger_freq': True, 'hess_diag_norm': None, 'hess_diag_cv': True, 'regret': True, 'tvi': True, 'sce': None},
                    opts={"AdamW": "AdamW", "Evie": "Evie", "EvieAmplify": "Evie-Amplify"}),
    "language": dict(unit_kind="shard", primary="valloss", higher=False, keys=['valloss', 'ppl', 'top1acc', 'steps50%', 'stepseps', 'wall(s)', 'peakmemmb', 'gflops', 'exaflops', 'stepsize', 'clipfreq%', 'clipcoord%', 'hessnorm', 'hesscv', 'regret'],
                     lower={'valloss': True, 'ppl': True, 'top1acc': False, 'steps50%': True, 'stepseps': True, 'wall(s)': True, 'peakmemmb': True, 'gflops': True, 'exaflops': True, 'stepsize': None, 'clipfreq%': True, 'clipcoord%': None, 'hessnorm': None, 'hesscv': True, 'regret': True},
                     opts={"AdamW": "AdamW", "Evie": "Evie", "EvieAmplify": "Evie-Amplify"}),
}


def per_unit(domain: str, ckpt: dict, opt: str) -> dict:
    """{unit: {metric: value}} where for finance/language the value is the median over the 3 seeds
    (exactly aggregate_ticker / aggregate_shard) and for vision each seed is a unit."""
    d, out = DOM[domain], {}
    for key, v in ckpt["raw"].items():
        o, unit = key.split("|", 1)
        if o != opt:
            continue
        if domain == "vision":
            out[unit] = {k: v.get(k, float("nan")) for k in d["keys"]}
        else:
            out[unit] = {k: (float(np.median([x for x in (r.get(k, float("nan")) for r in v) if not (isinstance(x, float) and math.isnan(x))] or [float("nan")]))) for k in d["keys"]}
    return out


def n_nan_runs(domain: str, ckpt: dict, opt: str) -> tuple:
    d = DOM[domain]
    tot = bad = 0
    for key, v in ckpt["raw"].items():
        if key.split("|", 1)[0] != opt:
            continue
        for r in ([v] if domain == "vision" else v):
            tot += 1
            bad += int(isinstance(r.get(d["primary"]), float) and math.isnan(r[d["primary"]]))
    return bad, tot


def vision_iterstotgt(ckpt_new: dict, ckpt_ref: dict) -> None:
    """iterstotgt is NaN in the AdaBelief copy (the target comes from AdamW runs): recompute offline with the
    target stored in the existing part-A checkpoint, by the same rule as compute_iters_to_target."""
    tgt = ckpt_ref.get("target_top1")
    if tgt is None:
        return
    for key, r in ckpt_new["raw"].items():
        r["iterstotgt"] = next((s for s, v in r.get("val_history", []) if v >= tgt), 5000)


def compare(domain: str, new_ck: dict, ref: dict, ref_opt: str, new_opt: str = "AdaBelief") -> dict:
    d = DOM[domain]
    a, b = per_unit(domain, new_ck, new_opt), per_unit(domain, ref, ref_opt)
    units = sorted(set(a) & set(b))
    rows, praw = {}, {}
    for k in d["keys"]:
        x = np.array([a[u][k] for u in units], float)
        y = np.array([b[u][k] for u in units], float)
        ok = np.isfinite(x) & np.isfinite(y)
        if ok.sum() == 0:
            rows[k] = dict(mean_new=float("nan"), sd_new=float("nan"), mean_ref=float("nan"), sd_ref=float("nan"), delta_pct=float("nan"), p=float("nan"), n_pairs=0)
            praw[k] = float("nan")
            continue
        lb = d["lower"].get(k)
        bm, nm = float(np.median(y[ok])), float(np.median(x[ok]))
        rel = float("nan") if bm == 0 else (bm - nm) / abs(bm) * 100.0
        delta = float("nan") if lb is None else (rel if lb else -rel)       # positive = AdaBelief better
        p = paired_wilcoxon_exact(x[ok], y[ok])["p"] if ok.sum() >= 3 else float("nan")
        praw[k] = p
        rows[k] = dict(mean_new=float(x[ok].mean()), sd_new=float(x[ok].std()), mean_ref=float(y[ok].mean()), sd_ref=float(y[ok].std()),
                       delta_pct=delta, p=p, n_pairs=int(ok.sum()))
    ph = holm({k: v for k, v in praw.items()})
    for k in rows:
        rows[k]["p_holm"] = ph.get(k, float("nan"))
    return rows


def fmt_p(domain: str, r: dict) -> str:
    if r["n_pairs"] < 3:
        return "n/a"
    if domain == "vision":
        return f"n={r['n_pairs']}: no significance claim possible (min exact p = {min_exact_p(r['n_pairs']):.3g})"
    return f"{r['p_holm']:.3e}"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", required=True, choices=list(DOM))
    ap.add_argument("--adabelief", required=True, help="AdaBelief checkpoint JSON")
    ap.add_argument("--existing", nargs="+", required=True, help="LABEL=path of existing part checkpoints, e.g. A=... B=... C=...")
    ap.add_argument("--out", default="tables")
    a = ap.parse_args(argv)
    d = DOM[a.domain]
    new = json.load(open(a.adabelief))
    refs = {lab: json.load(open(p)) for lab, p in (x.split("=", 1) for x in a.existing)}
    if a.domain == "vision" and "A" in refs:
        vision_iterstotgt(new, refs["A"])
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    md = [f"# AdaBelief vs existing columns ({a.domain})", "",
          f"Primary metric (pre-declared): {d['primary']} ({'higher' if d['higher'] else 'lower'} is better). Positive delta% = AdaBelief better.", ""]
    bad, tot = n_nan_runs(a.domain, new, "AdaBelief")
    md.append(f"AdaBelief runs: {tot}, NaN primary: {bad}. sessions: {len(new.get('sessions', []))}; permanent failures: {sorted(new.get('failed', {}))[:10]}")
    csv = ["ref,ref_optimizer,metric,mean_adabelief,sd_adabelief,mean_ref,sd_ref,delta_pct,p_raw,p_holm,n_pairs"]
    adamw_cols = {}
    for lab, ck in refs.items():
        for name, legacy in d["opts"].items():
            if not any(k.startswith(legacy + "|") for k in ck["raw"]):
                continue
            rows = compare(a.domain, new, ck, legacy)
            if name == "AdamW":
                adamw_cols[lab] = per_unit(a.domain, ck, legacy)
            md += ["", f"## vs {name} (part {lab})", "", "| metric | AdaBelief | reference | delta% | p_holm | n_pairs |", "|---|---|---|---|---|---|"]
            for k, r in rows.items():
                md.append(f"| {k} | {r['mean_new']:.4g}±{r['sd_new']:.2g} | {r['mean_ref']:.4g}±{r['sd_ref']:.2g} | {r['delta_pct']:+.2f}% | {fmt_p(a.domain, r)} | {r['n_pairs']} |")
                csv.append(f"{lab},{name},{k},{r['mean_new']},{r['sd_new']},{r['mean_ref']},{r['sd_ref']},{r['delta_pct']},{r['p']},{r['p_holm']},{r['n_pairs']}")
    labs = sorted(adamw_cols)
    if len(labs) >= 2:
        md += ["", "## Noise floor: median %Δ between the AdamW columns of different parts (same baseline, different run)", ""]
        for i in range(len(labs)):
            for j in range(i + 1, len(labs)):
                x = adamw_cols[labs[i]]; y = adamw_cols[labs[j]]
                u = sorted(set(x) & set(y))
                if not u:
                    continue
                v1 = np.array([x[t][d["primary"]] for t in u], float); v2 = np.array([y[t][d["primary"]] for t in u], float)
                ok = np.isfinite(v1) & np.isfinite(v2)
                md.append(f"- {d['primary']}: AdamW({labs[i]}) vs AdamW({labs[j]}): median Δ = {(np.median(v2[ok]) - np.median(v1[ok])) / abs(np.median(v1[ok])) * 100:+.2f}% (n={int(ok.sum())})")
    (out / f"adabelief_{a.domain}.md").write_text("\n".join(md) + "\n")
    (out / f"adabelief_{a.domain}.csv").write_text("\n".join(csv) + "\n")
    print(f"wrote {out}/adabelief_{a.domain}.md|csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
