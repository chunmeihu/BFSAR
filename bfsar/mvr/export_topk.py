"""Load the coarse similarity matrix written by dual-path evaluation."""

from __future__ import annotations

import numpy as np


def load_coarse_sim_npz(path: str):
    d = np.load(path, allow_pickle=True)
    return (
        d["sim"].astype(np.float32),
        list(d["refs"]),
        list(d["tgts"]),
        list(d["index_names"]),
    )
