"""E0a: audit that every inline optimizer copy in the run scripts equals the reference copy (vision/benchmark/run1_partA.py) and extract constants.
Reads files only (ast); trains nothing; edits nothing.  Output: audit_report.md (+ audit_report.json).
  python -m evie_exp.scripts.audit.audit_legacy_optimizers --repo . --out results/audit
Levels per class: IDENTICAL (normalised source equal), IDENTICAL-AFTER-LOGGING-STRIP (equal once log/print calls and docstrings are removed),
DIFFERS (unified diff of the normalised lines is listed: this is the signal that the gate formula or calibration may differ)."""
from __future__ import annotations

import argparse
import ast
import difflib
import json
from pathlib import Path

CLASSES = ("EvieOptimizer", "EvieAmplifyOptimizer")
CONSTS = ("TARGET_WP", "WELL_POS_MAX", "FAST_EMA_BETA", "CALIBRATE_STEP", "EVAL_EVERY", "MAX_STEPS", "BATCH_SIZE_DEFAULT", "BATCH_SIZE", "SEEDS",
          "OPT_NAMES", "LR_GRID", "LR_SEARCH_STEPS", "WEIGHT_DECAY", "GRAD_CLIP", "CLIP_NORM")
LOGNAMES = {"log", "print", "_log", "logger", "logging"}


class _Strip(ast.NodeTransformer):
    def __init__(self, strip_logs: bool):
        self.strip_logs = strip_logs

    def _body(self, node):
        body = []
        for i, st in enumerate(node.body):
            if i == 0 and isinstance(st, ast.Expr) and isinstance(getattr(st, "value", None), ast.Constant) and isinstance(st.value.value, str):
                continue
            if self.strip_logs and isinstance(st, ast.Expr) and isinstance(st.value, ast.Call):
                f = st.value.func
                nm = f.id if isinstance(f, ast.Name) else (f.attr if isinstance(f, ast.Attribute) else "")
                if nm in LOGNAMES:
                    continue
            body.append(self.visit(st))
        node.body = body or [ast.Pass()]
        return node

    visit_FunctionDef = visit_ClassDef = visit_If = visit_For = visit_With = visit_While = visit_Try = _body

    def generic_visit(self, node):
        if hasattr(node, "body") and isinstance(node.body, list) and not isinstance(node, ast.Module):
            return self._body(node)
        return super().generic_visit(node)


def norm(src_node: ast.AST, strip_logs: bool) -> list:
    import copy
    n = _Strip(strip_logs).visit(copy.deepcopy(src_node))
    ast.fix_missing_locations(n)
    return [l.rstrip() for l in ast.unparse(n).splitlines() if l.strip()]


def scan(path: Path):
    try:
        tree = ast.parse(path.read_text(errors="replace"))
    except SyntaxError as e:
        return None, None, str(e)
    classes, consts = {}, {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name in CLASSES:
            classes[node.name] = node
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name) and node.targets[0].id in CONSTS:
            consts[node.targets[0].id] = ast.unparse(node.value)
    return classes, consts, None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=".")
    ap.add_argument("--out", default="results/audit")
    ap.add_argument("--reference", default="vision/benchmark/run1_partA.py")
    a = ap.parse_args(argv)
    repo = Path(a.repo)
    files = sorted(p for d in ("vision", "finance", "language") for p in (repo / d).rglob("*.py")) + ([repo / "ablation.py"] if (repo / "ablation.py").exists() else [])
    ref_cls, _, err = scan(repo / a.reference)
    if not ref_cls:
        raise SystemExit(f"reference file {a.reference} unreadable or has no classes: {err}")
    ref = {k: (norm(v, False), norm(v, True)) for k, v in ref_cls.items()}
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    md = ["# E0a audit of inline optimizer copies", "", f"Reference: `{a.reference}` (classes {', '.join(ref)}). Files scanned: {len(files)}. No file was edited.", ""]
    rep, differing, none_found = {}, [], []
    for f in files:
        rel = str(f.relative_to(repo))
        cl, consts, err = scan(f)
        if err:
            md += [f"## {rel}", f"unparseable: {err}", ""]
            rep[rel] = {"error": err}
            continue
        if not cl:
            none_found.append(rel)
            continue
        md += [f"## {rel}", "", "| class | result |", "|---|---|"]
        rep[rel] = {"classes": {}, "constants": consts}
        diffs = []
        for name, node in cl.items():
            if name not in ref:
                continue
            full, stripped = norm(node, False), norm(node, True)
            if full == ref[name][0]:
                res = "IDENTICAL"
            elif stripped == ref[name][1]:
                res = "IDENTICAL-AFTER-LOGGING-STRIP"
            else:
                res = "DIFFERS"
                differing.append(f"{rel}:{name}")
                diffs.append((name, list(difflib.unified_diff(ref[name][1], stripped, "reference", rel, lineterm="", n=1))))
            rep[rel]["classes"][name] = res
            md.append(f"| {name} | {res} |")
        md += ["", "constants: " + (", ".join(f"`{k}={v}`" for k, v in consts.items()) or "none found"), ""]
        for name, d in diffs:
            md += [f"<details><summary>diff {name}</summary>", "", "```diff", *d[:200], "```", "</details>", ""]
    md += ["## Summary", "", f"- files with an inline copy: {len(rep)}", f"- files with NO inline copy (not audited): {len(none_found)}",
           f"- classes that differ from the reference after stripping logging/docstrings: {len(differing)}"]
    md += [f"  - `{d}`" for d in differing]
    md += ["", "Files where the gate formula or calibration differs: " + (", ".join(sorted({d.split(':')[0] for d in differing})) or "none")]
    (out / "audit_report.md").write_text("\n".join(md) + "\n")
    (out / "audit_report.json").write_text(json.dumps({"differing": differing, "files": rep, "no_inline_copy": none_found}, indent=1))
    print(f"[E0a] {len(rep)} files with inline copies, {len(differing)} differing classes -> {out/'audit_report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
