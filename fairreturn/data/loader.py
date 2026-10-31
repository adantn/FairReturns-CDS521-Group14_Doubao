"""Read the raw Online Retail II data from Excel or CSV and normalise the columns.

Accepted inputs in data/raw/ (any mix; exact duplicates are removed later):
  * online_retail_1.xlsx + online_retail_2.xlsx   (one sheet each, as split by the team)
  * online_retail_II.xlsx                          (original UCI workbook with two sheets)
  * online_retail_II.csv                           (Kaggle: cgrymn/online-retail-ii-uci-dataset)

Reading ~1M Excel rows takes 1-2 minutes, so the normalised table is cached in data/interim/.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from fairreturn import config as cfg

COLUMN_ALIASES = {
    "invoice": ["Invoice", "InvoiceNo", "invoice", "invoice_no"],
    "stock_code": ["StockCode", "Stock Code", "stock_code"],
    "description": ["Description", "description"],
    "quantity": ["Quantity", "quantity"],
    "invoice_date": ["InvoiceDate", "Invoice Date", "invoice_date"],
    "price": ["Price", "UnitPrice", "Unit Price", "price"],
    "customer_id": ["Customer ID", "CustomerID", "Customer Id", "customer_id"],
    "country": ["Country", "country"],
}
STANDARD_COLUMNS = list(COLUMN_ALIASES)
CACHE_FILE = cfg.DATA_INTERIM / "transactions.pkl"
CACHE_MANIFEST = cfg.DATA_INTERIM / "transactions_manifest.json"


def find_raw_files(raw_dir: Path | None = None) -> list[Path]:
    raw_dir = Path(raw_dir or cfg.DATA_RAW)
    files = sorted(p for p in raw_dir.iterdir()
                   if p.is_file() and p.suffix.lower() in (".xlsx", ".xls", ".csv") and not p.name.startswith("~$"))
    if not files:
        raise FileNotFoundError(
            f"No .xlsx or .csv files found in {raw_dir}. Copy online_retail_1.xlsx and online_retail_2.xlsx "
            "(or online_retail_II.csv from Kaggle) into that folder.")
    return files


def _rename(df: pd.DataFrame, source: str) -> pd.DataFrame:
    mapping = {}
    for std, aliases in COLUMN_ALIASES.items():
        for a in aliases:
            if a in df.columns:
                mapping[a] = std
                break
        else:
            raise ValueError(f"{source}: missing column for '{std}' (looked for {aliases}); found {list(df.columns)}")
    return df.rename(columns=mapping)[STANDARD_COLUMNS]


def normalise(df: pd.DataFrame) -> pd.DataFrame:
    """Consistent dtypes: strings stripped, numbers numeric, dates parsed, customer id as text."""
    out = pd.DataFrame()
    out["invoice"] = df["invoice"].astype(str).str.strip()
    out["stock_code"] = df["stock_code"].astype(str).str.strip()
    out["description"] = df["description"].fillna("").astype(str).str.strip().str.upper()
    out["quantity"] = pd.to_numeric(df["quantity"], errors="coerce").fillna(0).astype(np.int64)
    out["invoice_date"] = pd.to_datetime(df["invoice_date"], errors="coerce")
    out["price"] = pd.to_numeric(df["price"], errors="coerce").fillna(0.0).astype(float)
    cid = pd.to_numeric(df["customer_id"], errors="coerce")
    out["customer_id"] = cid.map(lambda v: "" if pd.isna(v) else str(int(v)))
    out["country"] = df["country"].fillna("Unspecified").astype(str).str.strip()
    out = out[out["invoice_date"].notna()]
    return out.reset_index(drop=True)


def read_file(path: Path) -> pd.DataFrame:
    t0 = time.time()
    if path.suffix.lower() in (".xlsx", ".xls"):
        sheets = pd.read_excel(path, sheet_name=None, dtype={"Invoice": str, "InvoiceNo": str, "StockCode": str})
        parts = [_rename(s, f"{path.name}:{name}") for name, s in sheets.items() if len(s)]
        df = pd.concat(parts, ignore_index=True)
    else:
        try:
            df = pd.read_csv(path, dtype={"Invoice": str, "InvoiceNo": str, "StockCode": str}, encoding="utf-8")
        except UnicodeDecodeError:
            df = pd.read_csv(path, dtype={"Invoice": str, "InvoiceNo": str, "StockCode": str}, encoding="ISO-8859-1")
        df = _rename(df, path.name)
    df = normalise(df)
    print(f"  read {path.name}: {len(df):,} rows in {time.time() - t0:.0f}s")
    return df


def _manifest(files: list[Path]) -> dict:
    return {"files": [{"name": p.name, "size": p.stat().st_size, "mtime": int(p.stat().st_mtime)} for p in files]}


def load_transactions(raw_dir: Path | None = None, use_cache: bool = True) -> pd.DataFrame:
    """All raw rows from every file in data/raw, normalised (duplicates NOT yet removed)."""
    files = find_raw_files(raw_dir)
    manifest = _manifest(files)
    if use_cache and CACHE_FILE.exists() and CACHE_MANIFEST.exists():
        try:
            if json.loads(CACHE_MANIFEST.read_text(encoding="utf-8")) == manifest:
                df = pd.read_pickle(CACHE_FILE)
                print(f"  loaded cached transactions: {len(df):,} rows ({CACHE_FILE.name})")
                return df
        except Exception as e:  # stale or incompatible cache -> rebuild
            print(f"  cache unusable ({e}); re-reading raw files")
    print(f"Reading {len(files)} raw file(s) from {files[0].parent} (Excel takes 1-2 minutes) ...")
    df = pd.concat([read_file(p) for p in files], ignore_index=True)
    cfg.DATA_INTERIM.mkdir(parents=True, exist_ok=True)
    df.to_pickle(CACHE_FILE)
    CACHE_MANIFEST.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return df
