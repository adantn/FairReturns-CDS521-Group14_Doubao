"""Turn C-invoices into a per-line return label.

Each cancellation line (customer c, product s, time t) is matched to the most recent earlier purchase
line of the same customer and product (pandas.merge_asof). A purchase line is labelled returned = 1 if
at least one matched cancellation happens within HORIZON_DAYS. Lines bought in the last HORIZON_DAYS
of the data are 'censored' (their return window has not closed) and are excluded from modelling.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from fairreturn import config as cfg


def label_returns(purchases: pd.DataFrame, cancellations: pd.DataFrame, data_end: pd.Timestamp,
                  horizon_days: int = cfg.HORIZON_DAYS) -> tuple[pd.DataFrame, dict]:
    p = purchases[["line_id", "customer_id", "stock_code", "invoice_date"]].sort_values("invoice_date")
    c = (cancellations[["customer_id", "stock_code", "invoice_date", "quantity"]]
         .rename(columns={"invoice_date": "cancel_date", "quantity": "cancel_qty"})
         .sort_values("cancel_date"))
    m = pd.merge_asof(c, p, left_on="cancel_date", right_on="invoice_date",
                      by=["customer_id", "stock_code"], direction="backward")
    matched = m[m["line_id"].notna()].copy()
    matched["line_id"] = matched["line_id"].astype(np.int64)
    matched["lag_days"] = (matched["cancel_date"] - matched["invoice_date"]).dt.total_seconds() / 86400.0
    within = matched[matched["lag_days"] <= horizon_days]
    per_line = within.groupby("line_id").agg(return_lag_days=("lag_days", "min"),
                                              returned_qty=("cancel_qty", lambda s: int(-s.sum())))

    out = purchases.copy()
    out = out.join(per_line, on="line_id")
    out["returned"] = out["return_lag_days"].notna().astype(np.int8)
    out["returned_qty"] = out["returned_qty"].fillna(0).astype(np.int64)
    cutoff = data_end.normalize() - pd.Timedelta(days=horizon_days)
    out["censored"] = out["invoice_date"] >= cutoff

    lags = matched["lag_days"]
    report = {
        "horizon_days": int(horizon_days),
        "cancellations_matched_share": round(float(m["line_id"].notna().mean()), 4) if len(m) else 0.0,
        "lag_days_median": round(float(lags.median()), 2) if len(lags) else None,
        "lag_days_p75": round(float(lags.quantile(0.75)), 2) if len(lags) else None,
        "share_matched_within_horizon": round(float((lags <= horizon_days).mean()), 4) if len(lags) else None,
        "same_day_share": round(float((lags < 1).mean()), 4) if len(lags) else None,
        "label_cutoff": str(cutoff.date()),
        "return_rate_uncensored": round(float(out.loc[~out["censored"], "returned"].mean()), 5),
    }
    return out, report
