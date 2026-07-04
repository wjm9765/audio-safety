"""Deterministic seeding across random, numpy, and (if installed) torch."""

import random

import numpy as np


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:  # torch lives in the gpu dependency group; stats-only envs run without it
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
