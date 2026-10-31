"""Small, self-contained sample of real test invoices for the web app (so it does not load 780k lines).

    python -m fairreturn.backend.demo_bundle

Picks every non-UK test invoice with a return, plus random UK and non-UK invoices with and without
returns (about 400 invoices). Writes outputs/demo/invoice_lines.pkl and outputs/demo/invoices.csv.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from fairreturn import config as cfg


def build(n_uk: int = 160, n_other: int = 140, seed: int = cfg.SEED) -> pd.DataFrame:
    L = pd.read_pickle(cfg.LINES_FILE)
    test = L[L["split"] == "test"]
    inv = test.groupby("invoice").agg(group=("group", "first"), country=("country", "first"),
                                      customer_id=("customer_id", "first"), date=("invoice_date", "first"),
                                      lines=("line_id", "size"), value=("line_value", "sum"),
                                      returned_lines=("returned", "sum"), postage=("has_postage", "max"))
    inv["has_return"] = (inv["returned_lines"] > 0).astype(int)
    rng = np.random.default_rng(seed)
    chosen = set(inv[(inv["group"] != "United Kingdom") & (inv["has_return"] == 1)].index)
    for uk, n in ((True, n_uk), (False, n_other)):
        for has_ret in (1, 0):
            pool = inv[((inv["group"] == "United Kingdom") == uk) & (inv["has_return"] == has_ret)
                       & (inv["lines"] <= 60)].index.difference(list(chosen))
            k = min(len(pool), n // 2)
            chosen.update(rng.choice(pool.to_numpy(), size=k, replace=False).tolist())
    inv = inv.loc[sorted(chosen)].reset_index().sort_values("date")
    lines = test[test["invoice"].isin(inv["invoice"])].copy()
    cfg.DEMO_DIR.mkdir(parents=True, exist_ok=True)
    lines.to_pickle(cfg.DEMO_DIR / "invoice_lines.pkl")
    inv.to_csv(cfg.DEMO_DIR / "invoices.csv", index=False)
    print(f"  demo bundle: {len(inv)} invoices, {len(lines):,} lines -> {cfg.DEMO_DIR}")
    return inv


if __name__ == "__main__":
    build()
