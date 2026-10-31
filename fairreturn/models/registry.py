"""All compared methods in one place, plus saving and loading of trained artifacts.

Each trained model is stored in outputs/models/<name>/:
    model.npz | model.joblib   weights
    features.json              the fitted FeatureBuilder (standardisation, one-hot levels)
    reference.npy              mean encoded training row (baseline for explanations)
    info.json                  method spec, training time, headline test metrics
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from fairreturn import config as cfg
from fairreturn.data.features import FeatureBuilder


@dataclass
class Method:
    name: str
    label: str
    family: str            # 'sklearn' | 'vibgrl'
    feature_set: str       # see data.features.FEATURE_SETS
    role: str              # 'baseline' | 'fairness baseline' | 'reference' | 'ablation' | 'ours' | 'variant'
    params: dict = field(default_factory=dict)
    deterministic: bool = False

    def spec(self) -> dict:
        return asdict(self)


METHODS: list[Method] = [
    # naive: the country is an input
    Method("lr_aware", "LR (country-aware)", "sklearn", "aware", "baseline", {"kind": "lr"}, True),
    Method("gbdt_aware", "GBDT (country-aware)", "sklearn", "aware", "baseline", {"kind": "gbdt"}, True),
    # "fairness through unawareness": the country column is dropped, the postage proxy stays
    Method("lr_unaware", "LR-Unaware (country column removed)", "sklearn", "unaware", "baseline",
           {"kind": "lr"}, True),
    Method("gbdt_unaware", "GBDT-Unaware (country column removed)", "sklearn", "unaware", "baseline",
           {"kind": "gbdt"}, True),
    # same inputs as our model: country and postage (set by the destination) removed
    Method("gbdt_blind", "GBDT-Blind (country and postage removed)", "sklearn", "blind", "baseline",
           {"kind": "gbdt"}, True),
    Method("gbdt_reweigh", "GBDT-Blind + Reweighing", "sklearn", "blind", "fairness baseline",
           {"kind": "gbdt", "reweigh": True}, True),
    # ablations of our model (same inputs, same network)
    Method("mlp", "MLP (no bottleneck, no adversary)", "vibgrl", "blind", "ablation", {"beta": 0.0, "lambda_max": 0.0}),
    Method("vib", "VIB only", "vibgrl", "blind", "ablation", {"beta": 1e-3, "lambda_max": 0.0}),
    Method("grl", "GRL only (EO)", "vibgrl", "blind", "ablation", {"beta": 0.0, "lambda_max": 1.0}),
    Method("vib_grl", "VIB + GRL (EO, ours)", "vibgrl", "blind", "ours", {"beta": 1e-3, "lambda_max": 1.0}),
    Method("vib_grl_dp", "VIB + GRL (DP variant)", "vibgrl", "blind", "variant",
           {"beta": 1e-3, "lambda_max": 1.0, "condition_on_y": False}),
    # what happens if the postage proxy is left in: the adversary makes the encoder use it in reverse
    Method("vib_grl_proxy", "VIB + GRL with the postage proxy kept", "vibgrl", "unaware", "variant",
           {"beta": 1e-3, "lambda_max": 1.0}),
    Method("gbdt_notags", "GBDT-Blind without product tags", "sklearn", "notags", "ablation",
           {"kind": "gbdt"}, True),
]
BY_NAME = {m.name: m for m in METHODS}


def get_method(name: str) -> Method:
    if name not in BY_NAME:
        raise KeyError(f"unknown method {name!r}; choose from {list(BY_NAME)}")
    return BY_NAME[name]


def build_model(method: Method, input_dim: int, n_groups: int, seed: int, epochs: int | None = None,
                overrides: dict | None = None):
    params = dict(method.params, **(overrides or {}))
    if method.family == "sklearn":
        from fairreturn.models.baselines import SklearnModel
        kind = params.pop("kind")
        reweigh = params.pop("reweigh", False)
        return SklearnModel(kind, seed=seed, reweigh=reweigh, **params)
    if method.family == "torch":
        from fairreturn.models.vib_grl_torch import TorchFairReturnModel
        return TorchFairReturnModel(input_dim, n_groups, seed=seed, **({"epochs": epochs} if epochs else {}), **params)
    from fairreturn.models.vib_grl import FairReturnModel
    if epochs:
        params["epochs"] = epochs
    return FairReturnModel(input_dim, n_groups, seed=seed, **params)


def model_dir(name: str, root: Path | None = None) -> Path:
    return Path(root or cfg.MODELS_DIR) / name


def save_artifacts(name: str, model, fb: FeatureBuilder, reference: np.ndarray, info: dict,
                   root: Path | None = None) -> Path:
    d = model_dir(name, root)
    d.mkdir(parents=True, exist_ok=True)
    fname = "model.joblib" if info.get("family") == "sklearn" else "model.npz"
    model.save(d / fname)
    fb.save(d / "features.json")
    np.save(d / "reference.npy", reference.astype(np.float32))
    (d / "info.json").write_text(json.dumps(info, indent=2, default=float), encoding="utf-8")
    return d


class LoadedModel:
    """A trained model bundled with its feature builder: predict straight from a derived-feature frame."""

    def __init__(self, name: str, root: Path | None = None):
        d = model_dir(name, root)
        if not d.exists():
            raise FileNotFoundError(f"{d} not found - run `python -m fairreturn.experiments.run_all` first")
        self.name = name
        self.info = json.loads((d / "info.json").read_text(encoding="utf-8"))
        self.fb = FeatureBuilder.load(d / "features.json")
        self.reference = np.load(d / "reference.npy")
        if (d / "model.joblib").exists():
            from fairreturn.models.baselines import SklearnModel
            self.model = SklearnModel.load(d / "model.joblib")
        else:
            from fairreturn.models.vib_grl import FairReturnModel
            self.model = FairReturnModel.load(d / "model.npz")
        self.label = self.info.get("label", name)

    def matrix(self, derived) -> np.ndarray:
        return self.fb.transform(derived)

    def predict(self, derived) -> np.ndarray:
        return self.model.predict_proba(self.matrix(derived))

    def predict_matrix(self, X) -> np.ndarray:
        return self.model.predict_proba(X)
