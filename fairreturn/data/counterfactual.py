"""Counterfactual 'twins': the same order line as if the customer lived in another country group.

Only the country and the attributes that are set by the destination change:
  * group / country             -> a different group (country sampled from that group's mix)
  * has_postage (POST / C2 line) -> resampled from the target group's postage rate
Everything describing the customer's behaviour, the product and the basket stays identical, and the
true label is unchanged. If a model's decision flips for the twin, the decision depended on country.
The whole invoice moves together, so all lines of one invoice get the same target country.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from fairreturn import config as cfg


def group_postage_rates(invoices: pd.DataFrame) -> dict[str, float]:
    rates = invoices.groupby("group")["has_postage"].mean().to_dict()
    return {g: float(rates.get(g, 0.0)) for g in cfg.GROUPS}


def group_country_weights(invoices: pd.DataFrame) -> dict[str, dict[str, float]]:
    out = {}
    for g in cfg.GROUPS:
        vc = invoices.loc[invoices["group"] == g, "country"].value_counts(normalize=True)
        out[g] = {str(k): float(v) for k, v in vc.items()} or {g: 1.0}
    return out


def make_twins(lines: pd.DataFrame, postage_rates: dict[str, float],
               country_weights: dict[str, dict[str, float]], seed: int = cfg.SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    inv = lines[["invoice", "group"]].drop_duplicates("invoice").reset_index(drop=True)
    groups = np.array(cfg.GROUPS)
    own = inv["group"].map({g: i for i, g in enumerate(cfg.GROUPS)}).to_numpy()
    shift = rng.integers(1, len(groups), size=len(inv))
    target = groups[(own + shift) % len(groups)]
    countries = []
    for g in target:
        w = country_weights[g]
        names = list(w)
        countries.append(names[rng.choice(len(names), p=np.array(list(w.values())) / sum(w.values()))])
    postage = (rng.random(len(inv)) < np.array([postage_rates[g] for g in target])).astype(np.int8)
    plan = pd.DataFrame({"invoice": inv["invoice"], "twin_group": target, "twin_country": countries,
                         "twin_postage": postage}).set_index("invoice")

    tw = lines.copy()
    p = plan.loc[tw["invoice"]]
    tw["twin_of"] = tw["line_id"].to_numpy()
    tw["group"] = p["twin_group"].to_numpy()
    tw["country"] = p["twin_country"].to_numpy()
    tw["has_postage"] = p["twin_postage"].to_numpy().astype(float)
    tw["is_twin"] = 1
    return tw.reset_index(drop=True)
