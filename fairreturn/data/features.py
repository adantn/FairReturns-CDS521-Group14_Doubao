"""Point-in-time features and the feature matrix builder.

Two layers, shared by the offline pipeline and the web API so both compute features identically:

  1. *Base columns* - human-readable raw facts about an order line, e.g. quantity, the customer's number
     of earlier invoices, how often this product was returned before. In the pipeline they are computed
     from history strictly BEFORE the invoice day (no look-ahead); in the API they come from the form.
  2. *Derived columns* (finalize_features) - log transforms, smoothed rates, seasonality, etc.

FeatureBuilder then standardises numeric columns and one-hot encodes categorical ones for a chosen
feature set ('aware', 'unaware', 'blind', 'notags').
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from fairreturn import config as cfg

# ----------------------------------------------------------------------------- column lists
NUMERIC = [
    "log_qty", "log_price", "log_line_value", "price_rel", "line_share",
    "basket_lines_log", "basket_value_log", "basket_qty_log",
    "cust_prior_invoices_log", "cust_prior_lines_log", "cust_prior_returns_log", "cust_prior_return_rate",
    "cust_prior_spend_log", "cust_tenure_log", "cust_recency_log", "cust_is_new",
    "prod_prior_lines_log", "prod_prior_return_rate", "prod_is_new",
    "tag_fragile", "tag_seasonal",
    "month_sin", "month_cos", "hour",
]
PROXY = ["has_postage"]                      # set by the destination country -> country proxy
CATEGORICAL = ["tag_category", "tag_material", "dow"]
SENSITIVE = ["group"]
TAG_COLUMNS = ["tag_category", "tag_material", "tag_fragile", "tag_seasonal"]

FEATURE_SETS = {
    "aware": NUMERIC + PROXY + CATEGORICAL + SENSITIVE,     # what a naive shop would use
    "unaware": NUMERIC + PROXY + CATEGORICAL,               # country column removed, the postage proxy remains
    "blind": NUMERIC + CATEGORICAL,                         # country and postage removed: the inputs of our model
    "notags": [c for c in NUMERIC + CATEGORICAL if not c.startswith("tag_")],  # blind without GenAI tags (ablation)
}

FEATURE_GROUP_OF = {  # used for explanations in the API and for the LLM 'tool' evidence
    "Customer history": [c for c in NUMERIC if c.startswith("cust_")],
    "Product history": ["prod_prior_lines_log", "prod_prior_return_rate", "prod_is_new", "price_rel"],
    "Product attributes": ["tag_fragile", "tag_seasonal", "tag_category", "tag_material"],
    "Order & basket": ["log_qty", "log_price", "log_line_value", "line_share",
                       "basket_lines_log", "basket_value_log", "basket_qty_log"],
    "Timing": ["month_sin", "month_cos", "hour", "dow"],
    "Postage (country proxy)": ["has_postage"],
    "Country": ["group"],
}

BASE_COLUMNS = [
    "quantity", "price", "product_median_price", "basket_lines", "basket_value", "basket_qty", "has_postage",
    "cust_prior_invoices", "cust_prior_lines", "cust_prior_returns", "cust_prior_spend",
    "cust_tenure_days", "cust_recency_days", "prod_prior_lines", "prod_prior_returns",
    "invoice_date", "tag_category", "tag_material", "tag_fragile", "tag_seasonal", "group",
]


# ----------------------------------------------------------------------------- history (pipeline only)
def _prior_daily_counts(purchases: pd.DataFrame, cancellations: pd.DataFrame, key: str) -> pd.DataFrame:
    """Cumulative counts per key strictly before each day (index: key, day)."""
    p = purchases.assign(day=purchases["invoice_date"].dt.normalize())
    c = cancellations.assign(day=cancellations["invoice_date"].dt.normalize())
    pd_ = p.groupby([key, "day"]).agg(p_lines=("line_id", "size"), p_inv=("invoice", "nunique"),
                                      p_value=("line_value", "sum"))
    cd_ = c.groupby([key, "day"]).size().rename("c_lines").to_frame()
    daily = pd_.join(cd_, how="outer").fillna(0.0).sort_index()
    cum = daily.groupby(level=0).cumsum()
    return cum - daily


def add_history(purchases: pd.DataFrame, cancellations: pd.DataFrame, invoices: pd.DataFrame,
                median_price: pd.Series) -> pd.DataFrame:
    """Attach base columns computed only from events before the invoice day."""
    out = purchases.copy()
    out["day"] = out["invoice_date"].dt.normalize()

    cust = _prior_daily_counts(purchases, cancellations, "customer_id")
    cust = cust.rename(columns={"p_lines": "cust_prior_lines", "p_inv": "cust_prior_invoices",
                                "p_value": "cust_prior_spend", "c_lines": "cust_prior_returns"})
    out = out.join(cust, on=["customer_id", "day"])

    days = out[["customer_id", "day"]].drop_duplicates().sort_values(["customer_id", "day"])
    days["prev_day"] = days.groupby("customer_id")["day"].shift(1)
    days["first_day"] = days.groupby("customer_id")["day"].transform("min")
    out = out.merge(days, on=["customer_id", "day"], how="left")
    out["cust_tenure_days"] = (out["day"] - out["first_day"]).dt.days.astype(float)
    out["cust_recency_days"] = (out["day"] - out["prev_day"]).dt.days.astype(float).fillna(365.0)

    prod = _prior_daily_counts(purchases, cancellations, "stock_code")[["p_lines", "c_lines"]]
    prod = prod.rename(columns={"p_lines": "prod_prior_lines", "c_lines": "prod_prior_returns"})
    out = out.join(prod, on=["stock_code", "day"])

    inv = invoices.set_index("invoice")[["basket_lines", "basket_value", "basket_qty", "has_postage"]]
    out = out.join(inv, on="invoice")
    out["product_median_price"] = out["stock_code"].map(median_price)
    for col in ["cust_prior_lines", "cust_prior_invoices", "cust_prior_spend", "cust_prior_returns",
                "prod_prior_lines", "prod_prior_returns"]:
        out[col] = out[col].fillna(0.0)
    return out.drop(columns=["day", "prev_day", "first_day"])


# ----------------------------------------------------------------------------- derived columns (shared)
def smoothed_rate(events, trials, alpha: float, prior: float):
    rate = (np.asarray(events, dtype=float) + alpha * prior) / (np.asarray(trials, dtype=float) + alpha)
    return np.clip(rate, 0.0, 1.0)


def finalize_features(base: pd.DataFrame, p0: float) -> pd.DataFrame:
    """Base columns -> model-ready columns (numeric + categorical, not yet encoded)."""
    b = base
    q = b["quantity"].astype(float).clip(lower=1)
    price = b["price"].astype(float).clip(lower=0.001)
    value = q * price
    basket_value = b["basket_value"].astype(float).clip(lower=0.001)
    med = pd.to_numeric(b["product_median_price"], errors="coerce")
    d = pd.DataFrame(index=b.index)
    d["log_qty"] = np.log1p(q)
    d["log_price"] = np.log1p(price)
    d["log_line_value"] = np.log1p(value)
    d["price_rel"] = np.where(med > 0, np.log(price) - np.log(med.where(med > 0, 1.0)), 0.0).clip(-3, 3)
    d["line_share"] = (value / basket_value).clip(0, 1)
    d["basket_lines_log"] = np.log1p(b["basket_lines"].astype(float))
    d["basket_value_log"] = np.log1p(basket_value)
    d["basket_qty_log"] = np.log1p(b["basket_qty"].astype(float))
    d["cust_prior_invoices_log"] = np.log1p(b["cust_prior_invoices"].astype(float))
    d["cust_prior_lines_log"] = np.log1p(b["cust_prior_lines"].astype(float))
    d["cust_prior_returns_log"] = np.log1p(b["cust_prior_returns"].astype(float))
    d["cust_prior_return_rate"] = smoothed_rate(b["cust_prior_returns"], b["cust_prior_lines"],
                                                cfg.SMOOTH_ALPHA_CUSTOMER, p0)
    d["cust_prior_spend_log"] = np.log1p(b["cust_prior_spend"].astype(float).clip(lower=0))
    d["cust_tenure_log"] = np.log1p(b["cust_tenure_days"].astype(float).clip(lower=0))
    d["cust_recency_log"] = np.log1p(b["cust_recency_days"].astype(float).clip(lower=0))
    d["cust_is_new"] = (b["cust_prior_invoices"].astype(float) <= 0).astype(float)
    d["prod_prior_lines_log"] = np.log1p(b["prod_prior_lines"].astype(float))
    d["prod_prior_return_rate"] = smoothed_rate(b["prod_prior_returns"], b["prod_prior_lines"],
                                                cfg.SMOOTH_ALPHA_PRODUCT, p0)
    d["prod_is_new"] = (b["prod_prior_lines"].astype(float) <= 0).astype(float)
    d["tag_fragile"] = b["tag_fragile"].astype(float)
    d["tag_seasonal"] = b["tag_seasonal"].astype(float)
    dt = pd.to_datetime(b["invoice_date"])
    month = dt.dt.month.astype(float)
    d["month_sin"] = np.sin(2 * np.pi * month / 12.0)
    d["month_cos"] = np.cos(2 * np.pi * month / 12.0)
    d["hour"] = dt.dt.hour.astype(float) + dt.dt.minute.astype(float) / 60.0
    d["has_postage"] = b["has_postage"].astype(float)
    d["tag_category"] = b["tag_category"].astype(str)
    d["tag_material"] = b["tag_material"].astype(str)
    d["dow"] = dt.dt.dayofweek.astype(int).astype(str)
    d["group"] = b["group"].astype(str)
    return d


# ----------------------------------------------------------------------------- encoder
class FeatureBuilder:
    """Standardise numeric columns, one-hot encode categoricals, for one feature set."""

    def __init__(self, feature_set: str = "unaware"):
        if feature_set not in FEATURE_SETS:
            raise ValueError(f"unknown feature set {feature_set!r}; choose from {list(FEATURE_SETS)}")
        self.feature_set = feature_set
        cols = FEATURE_SETS[feature_set]
        self.numeric = [c for c in cols if c not in CATEGORICAL + SENSITIVE]
        self.categorical = [c for c in cols if c in CATEGORICAL + SENSITIVE]
        self.mean: dict[str, float] = {}
        self.std: dict[str, float] = {}
        self.levels: dict[str, list[str]] = {}

    def fit(self, d: pd.DataFrame) -> "FeatureBuilder":
        for c in self.numeric:
            v = d[c].astype(float)
            self.mean[c] = float(v.mean())
            s = float(v.std())
            self.std[c] = s if s > 1e-9 else 1.0
        for c in self.categorical:
            self.levels[c] = list(cfg.GROUPS) if c == "group" else sorted(d[c].astype(str).unique().tolist())
        return self

    def transform(self, d: pd.DataFrame) -> np.ndarray:
        n = len(d)
        blocks = []
        if self.numeric:
            num = np.empty((n, len(self.numeric)), dtype=np.float32)
            for j, c in enumerate(self.numeric):
                num[:, j] = (d[c].astype(float).to_numpy() - self.mean[c]) / self.std[c]
            blocks.append(num)
        for c in self.categorical:
            vals = d[c].astype(str).to_numpy()
            blocks.append(np.stack([vals == v for v in self.levels[c]], axis=1).astype(np.float32))
        return np.concatenate(blocks, axis=1) if blocks else np.zeros((n, 0), dtype=np.float32)

    def fit_transform(self, d):
        return self.fit(d).transform(d)

    @property
    def feature_names(self) -> list[str]:
        return self.numeric + [f"{c}={v}" for c in self.categorical for v in self.levels[c]]

    @property
    def n_features(self) -> int:
        return len(self.feature_names)

    def column_groups(self) -> dict[str, list[int]]:
        """Explanation group -> column indices in the encoded matrix."""
        names = self.feature_names
        out = {}
        for g, cols in FEATURE_GROUP_OF.items():
            idx = [i for i, n in enumerate(names) if n.split("=")[0] in cols]
            if idx:
                out[g] = idx
        return out

    def to_dict(self) -> dict:
        return {"feature_set": self.feature_set, "numeric": self.numeric, "categorical": self.categorical,
                "mean": self.mean, "std": self.std, "levels": self.levels}

    @classmethod
    def from_dict(cls, dct: dict) -> "FeatureBuilder":
        fb = cls(dct["feature_set"])
        fb.numeric, fb.categorical = dct["numeric"], dct["categorical"]
        fb.mean, fb.std, fb.levels = dct["mean"], dct["std"], dct["levels"]
        return fb

    def save(self, path: Path):
        Path(path).write_text(json.dumps(self.to_dict(), indent=1), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "FeatureBuilder":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
