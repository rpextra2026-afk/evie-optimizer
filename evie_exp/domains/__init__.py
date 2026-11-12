from __future__ import annotations

import torch

from . import fake, finance, language, vision

DOMAINS = {"vision": vision, "finance": finance, "language": language}


def build_task(domain: str, unit: str, seed: int, device: torch.device, batch_size: int = 32, fake_data: bool = False):
    if fake_data:
        return fake.build_fake(domain, unit, seed, device, batch_size)
    return DOMAINS[domain].build(unit, seed, device, batch_size)
