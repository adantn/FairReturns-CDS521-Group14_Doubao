"""Baselines built on scikit-learn: logistic regression, gradient-boosted trees, and Reweighing.

Reweighing (Kamiran & Calders, 2012) is a classic pre-processing fairness method: each training row gets
weight P(group) * P(label) / P(group, label), so that group and label look independent to the learner.
"""
from __future__ import annotations

import io
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression


def reweighing_weights(y, a) -> np.ndarray:
    y = np.asarray(y).astype(int)
    a = np.asarray(a).astype(int)
    w = np.ones(len(y), dtype=float)
    n = len(y)
    for g in np.unique(a):
        for lab in (0, 1):
            m = (a == g) & (y == lab)
            if m.any():
                w[m] = ((a == g).sum() / n) * ((y == lab).sum() / n) / (m.sum() / n)
    return w


class SklearnModel:
    def __init__(self, kind: str = "gbdt", seed: int = 0, reweigh: bool = False, **params):
        self.kind, self.seed, self.reweigh, self.params = kind, seed, reweigh, params
        if kind == "lr":
            self.est = LogisticRegression(max_iter=params.get("max_iter", 2000), C=params.get("C", 1.0))
        elif kind == "gbdt":
            self.est = HistGradientBoostingClassifier(
                max_iter=params.get("max_iter", 300), learning_rate=params.get("learning_rate", 0.05),
                max_leaf_nodes=params.get("max_leaf_nodes", 31), min_samples_leaf=params.get("min_samples_leaf", 50),
                l2_regularization=params.get("l2_regularization", 1.0), early_stopping=False, random_state=seed)
        else:
            raise ValueError(f"unknown baseline kind {kind!r}")
        self.config = {"kind": kind, "seed": seed, "reweigh": reweigh, **params}
        self.history: dict = {}

    def fit(self, X, y, a=None, X_val=None, y_val=None, a_val=None, verbose: bool = False):
        w = reweighing_weights(y, a) if (self.reweigh and a is not None) else None
        self.est.fit(X, np.asarray(y).astype(int), sample_weight=w)
        return self

    def predict_proba(self, X) -> np.ndarray:
        return self.est.predict_proba(X)[:, 1]

    def save(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"est": self.est, "config": self.config}, path)

    @classmethod
    def load(cls, path) -> "SklearnModel":
        d = joblib.load(path)
        c = dict(d["config"])
        obj = cls(c.pop("kind"), c.pop("seed"), c.pop("reweigh"), **c)
        obj.est = d["est"]
        return obj

    def size_kb(self) -> float:
        buf = io.BytesIO()
        joblib.dump(self.est, buf)
        return len(buf.getvalue()) / 1024.0

    def n_params(self, inference_only: bool = True) -> int | None:
        if self.kind == "lr":
            return int(self.est.coef_.size + self.est.intercept_.size)
        return None
