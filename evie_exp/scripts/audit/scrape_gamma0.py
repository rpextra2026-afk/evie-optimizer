"""E0b: scrape gamma0 / fast_sigma2 from existing log files -> calibration-clip status per domain (theorem B4(iv)).  Reads files only.
  python -m evie_exp.scripts.audit.scrape_gamma0 --logs <dir of logs> --out results/audit
Matches e.g. '[Evie] gamma0=123.4  fast_sigma2=1.2e-03' or '[Evie-Amplify] ...'.  Vision classes log no gamma0 line (use the E2 diagnostics); if no line
is found the script says so and writes nothing invented."""
from __future__ import annotations

import argparse
import csv
import re
import statistics
from pathlib import Path

PAT = re.compile(r"(Evie-Amplify|Evie|Evie-Amplify|Evie)\W{0,3}\s+gamma0=([0-9.]+)\s+fast_sigma2=([0-9.eE+-]+)")
W = 0.85
LO, HI = (1 / W - 1) / 5000.0, (1 / W - 1) / 10.0


def domain_of(path: Path) -> str:
    s = str(path).lower()
    for d in ("vision", "finance", "language", "lm_"):
        if d in s:
            return "language" if d == "lm_" else d
    return "unknown"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", required=True)
    ap.add_argument("--out", default="results/audit")
    a = ap.parse_args(argv)
    rows = []
    files = [p for p in Path(a.logs).rglob("*") if p.is_file() and p.suffix.lower() in (".log", ".txt", ".out", ".md", "")]
    for p in files:
        try:
            txt = p.read_text(errors="replace")
        except OSError:
            continue
        for m in PAT.finditer(txt):
            rows.append((str(p), domain_of(p), m.group(1).replace("Evie", "Evie") if m.group(1).startswith("Evie") else m.group(1), float(m.group(2)), float(m.group(3))))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if not rows:
        (out / "gamma0_summary.md").write_text("# gamma0 scrape\n\nNo `gamma0=... fast_sigma2=...` lines were found under the given log directory. Nothing is inferred.\n")
        print(f"[E0b] no matching lines under {a.logs} (scanned {len(files)} files); nothing written but the note")
        return 0
    with open(out / "gamma0_by_run.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["file", "domain", "optimizer", "gamma0", "sigma_bar_f2"])
        w.writerows(rows)
    md = ["# gamma0 scrape (B4(iv): is the calibration clip [10, 5000] inactive?)", "", f"Clip-inactive window for sigma_bar_f^2 at w=0.85: [{LO:.4e}, {HI:.4e}]", "",
          "| domain | optimizer | n | min g0 | median g0 | max g0 | frac g0==10 | frac g0==5000 | frac sigma in window |", "|---|---|---|---|---|---|---|---|---|"]
    keys = sorted({(r[1], r[2]) for r in rows})
    for dom, opt in keys:
        g = [r[3] for r in rows if (r[1], r[2]) == (dom, opt)]
        s = [r[4] for r in rows if (r[1], r[2]) == (dom, opt)]
        md.append(f"| {dom} | {opt} | {len(g)} | {min(g):g} | {statistics.median(g):g} | {max(g):g} | {sum(x == 10.0 for x in g) / len(g):.3f} | "
                  f"{sum(x == 5000.0 for x in g) / len(g):.3f} | {sum(LO <= x <= HI for x in s) / len(s):.3f} |")
    md += ["", "Vision classes log no gamma0 line; vision clip status comes from the E2 diagnostics."]
    (out / "gamma0_summary.md").write_text("\n".join(md) + "\n")
    print(f"[E0b] {len(rows)} calibration lines -> {out/'gamma0_summary.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
