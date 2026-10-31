"""Clean raw transactions and split them into purchases, cancellations (returns) and invoice summaries.

Rules (each step is counted in the data report):
  1. drop exact duplicate rows (this also removes the 2010-12-01..09 overlap between the two sheets)
  2. drop rows without Customer ID (returns cannot be linked to a buyer)
  3. invoices starting with 'C' are cancellations / returns; 'A' are bad-debt adjustments (dropped)
  4. non-product stock codes (postage, fees, discounts, vouchers) are kept only to flag postage
  5. purchases: product lines with quantity > 0 and price > 0
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from fairreturn import config as cfg


def flag_rows(tx: pd.DataFrame) -> pd.DataFrame:
    tx = tx.copy()
    code = tx["stock_code"].str.upper()
    tx["is_cancel"] = tx["invoice"].str.upper().str.startswith("C")
    tx["is_adjust"] = tx["invoice"].str.upper().str.startswith("A")
    tx["is_service"] = code.isin(cfg.SERVICE_CODES) | code.str.startswith(cfg.SERVICE_PREFIXES)
    tx["is_postage"] = code.isin(cfg.POSTAGE_CODES)
    tx["group"] = tx["country"].map(cfg.country_group)
    return tx


def clean_transactions(tx: pd.DataFrame, drop_duplicates: bool = True) -> tuple[pd.DataFrame, dict]:
    report = {"raw_rows": int(len(tx)), "raw_customers": int(tx.loc[tx.customer_id != "", "customer_id"].nunique()),
              "date_min": str(tx["invoice_date"].min()), "date_max": str(tx["invoice_date"].max())}
    if drop_duplicates:
        before = len(tx)
        tx = tx.drop_duplicates(ignore_index=True)
        report["duplicate_rows_removed"] = int(before - len(tx))
    missing = tx["customer_id"] == ""
    report["rows_without_customer"] = int(missing.sum())
    report["share_without_customer"] = round(float(missing.mean()), 4)
    tx = tx[~missing].reset_index(drop=True)
    tx = flag_rows(tx)
    report["rows_with_customer"] = int(len(tx))
    report["customers"] = int(tx["customer_id"].nunique())
    report["countries"] = int(tx["country"].nunique())
    return tx, report


def split_events(tx: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """Return (purchases, cancellations, invoices, report)."""
    product = ~tx["is_service"] & ~tx["is_adjust"]
    purchases = tx[product & ~tx["is_cancel"] & (tx["quantity"] > 0) & (tx["price"] > 0)].copy()
    cancellations = tx[product & tx["is_cancel"] & (tx["quantity"] < 0)].copy()
    purchases["line_value"] = purchases["quantity"] * purchases["price"]
    purchases = purchases.sort_values(["invoice_date", "invoice"], kind="mergesort").reset_index(drop=True)
    purchases.insert(0, "line_id", np.arange(len(purchases), dtype=np.int64))
    cancellations = cancellations.reset_index(drop=True)

    postage_invoices = set(tx.loc[tx["is_postage"] & ~tx["is_cancel"], "invoice"])
    inv = purchases.groupby("invoice", sort=False).agg(
        invoice_date=("invoice_date", "first"), customer_id=("customer_id", "first"),
        country=("country", "first"), group=("group", "first"),
        basket_lines=("line_id", "size"), basket_value=("line_value", "sum"), basket_qty=("quantity", "sum"))
    inv["has_postage"] = inv.index.isin(postage_invoices).astype(np.int8)
    inv = inv.reset_index()

    report = {
        "purchase_lines": int(len(purchases)), "purchase_invoices": int(inv.shape[0]),
        "cancellation_lines": int(len(cancellations)), "cancellation_invoices": int(cancellations["invoice"].nunique()),
        "products": int(purchases["stock_code"].nunique()),
        "invoices_with_postage": int(inv["has_postage"].sum()),
    }
    return purchases, cancellations, inv, report
