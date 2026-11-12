"""Seeding and loader generators (fixes D-3: no hidden RNG consumption, so all arms of a unit
see identical init and identical batch order)."""
from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class Generators:
    init: torch.Generator
    loader: torch.Generator


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed % (2 ** 32))
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_generators(seed: int) -> Generators:
    """``init`` seeds the global RNG before model construction; ``loader`` drives batch order."""
    return Generators(init=torch.Generator().manual_seed(seed),
                      loader=torch.Generator().manual_seed(10_000 + seed))
