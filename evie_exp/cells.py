"""Three ablation cells: AV (vision), AF (finance), AL (language); profiles from configs/profiles."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

import yaml

from .domains import finance, language, vision

CONF = Path(__file__).resolve().parent / "configs"


@dataclass(frozen=True)
class Cell:
    name: str
    domain: str
    batch_size: int = 32
    steps: int = 5000
    eval_every: int = 100
    primary: str = "test_top1"
    higher_is_better: bool = True
    # conservative per-run second estimates used only until real timings exist
    est_run_s: float = 90.0


CELLS = {
    "AV": Cell("AV", "vision", eval_every=100, primary="test_top1", higher_is_better=True, est_run_s=90.0),
    "AF": Cell("AF", "finance", eval_every=100, primary="test_mse", higher_is_better=False, est_run_s=60.0),
    "AL": Cell("AL", "language", eval_every=250, primary="test_loss", higher_is_better=False, est_run_s=240.0),
}

BASE_CFG = {"weight_decay": 1e-4, "w_target": 0.85, "grad_clip_norm": 5.0, "cosine_eta_min_frac": 0.1,
            "calib_step": 125, "adabelief_eps": 1e-16}
CALIB_SEED = 9001


def load_profile(name: str) -> dict:
    p = CONF / "profiles" / f"{name}.yaml"
    if not p.exists():
        raise FileNotFoundError(f"unknown profile {name!r}; have {[x.stem for x in (CONF / 'profiles').glob('*.yaml')]}")
    return yaml.safe_load(open(p))


def eval_units(cell: Cell, max_units: int | None = None, fake: bool = False) -> List[str]:
    if fake:
        return [f"fake{i}" for i in range(max_units or 4)]
    if cell.domain == "vision":
        u = vision.units_for("eval", 10)
    elif cell.domain == "finance":
        u = finance.units_for("eval")
    else:
        u = language.units_for("eval")
    return u[:max_units] if max_units else u


def tune_units(cell: Cell, profile: dict | None = None) -> List[str]:
    if profile and profile.get("_fake"):
        return ["faket0", "faket1"]
    if cell.domain == "vision":
        return vision.units_for("tune", 3)
    if cell.domain == "finance":
        return finance.units_for("tune")
    return language.units_for("tune")


def eval_seeds(cell: Cell, profile: dict) -> List[int]:
    return list(profile["seeds_per_unit"][cell.domain])


def tune_seeds(cell: Cell, profile: dict) -> List[int]:
    return list(profile["tune_seeds"][cell.domain])


def tune_steps(cell: Cell, profile: dict) -> int:
    if profile.get("_smoke"):
        return 200
    return max(int(round(profile["tune_frac"] * cell.steps)), 200)


def cfg_for(cell: Cell, profile_name: str, profile: dict) -> dict:
    return {**BASE_CFG, "cell": cell.name, "profile": profile_name, "profile_content": profile}
