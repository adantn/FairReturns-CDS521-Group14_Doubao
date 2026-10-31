"""Model-agnostic explanations by occlusion of feature groups.

For each feature group (customer history, product history, product attributes, order and basket,
timing, postage, country) we replace that block of the encoded row by its training average and record
how much the predicted risk changes. Positive = this part of the order raises the risk relative to an
average order. Cheap (one extra prediction per group) and works for every model in the project.
"""
from __future__ import annotations

import numpy as np

from fairreturn.data.features import FeatureBuilder


def contributions(predict_matrix, X: np.ndarray, fb: FeatureBuilder, reference: np.ndarray) -> list[dict]:
    base = predict_matrix(X)
    groups = fb.column_groups()
    out = [dict() for _ in range(len(X))]
    for g, idx in groups.items():
        Xo = X.copy()
        Xo[:, idx] = reference[idx]
        delta = base - predict_matrix(Xo)
        for i in range(len(X)):
            out[i][g] = float(delta[i])
    return out


def top_factors(contrib: dict, k: int = 3, min_abs: float = 1e-4) -> list[tuple[str, float]]:
    items = sorted(contrib.items(), key=lambda kv: -abs(kv[1]))
    return [(g, v) for g, v in items[:k] if abs(v) >= min_abs]
