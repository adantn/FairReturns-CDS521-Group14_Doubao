"""Build the modelling dataset from the raw Online Retail II files.

    python -m fairreturn.data.prepare                 # rule-based product tags (default, offline)
    python -m fairreturn.data.prepare --tags llm      # use outputs/genai/product_tags_llm.csv where valid
    python -m fairreturn.data.prepare --no-cache      # re-read the Excel files

Writes data/processed/:
    lines.pkl                one row per purchase line: ids, base facts, model features, label, split
    twins_test.pkl           counterfactual twins of every test line (country group swapped)
    products.csv             product catalogue: description, popularity, typical price, tags, history
    customers_snapshot.csv   per-customer history at the end of the data (used by the API for new orders)
    product_tags_rules.csv   rule-based tags for every product
    meta.json                data report, split sizes, group statistics, smoothing prior, postage rates
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd

from fairreturn import config as cfg
from fairreturn.data import cleaning, counterfactual, features, labels, loader
from fairreturn.genai.product_tagger import tag_products_with_rules


def assign_splits(lines: pd.DataFrame) -> pd.Series:
    d = lines["invoice_date"]
    s = pd.Series("test", index=lines.index, dtype=object)
    s[d < pd.Timestamp(cfg.SPLIT_DATES["test_start"])] = "val"
    s[d < pd.Timestamp(cfg.SPLIT_DATES["val_start"])] = "train"
    s[d < pd.Timestamp(cfg.SPLIT_DATES["train_start"])] = "burnin"
    s[lines["censored"]] = "censored"
    return s


def build_products(lines: pd.DataFrame, cancellations: pd.DataFrame, median_price: pd.Series,
                   tag_mode: str) -> pd.DataFrame:
    usable = lines[lines["split"].isin(["train", "val", "test"])]
    desc = (lines.groupby("stock_code")["description"]
            .agg(lambda s: s.value_counts().index[0] if len(s) else ""))
    prod = pd.DataFrame({"description": desc})
    prod["n_lines"] = lines.groupby("stock_code").size()
    prod["n_lines_usable"] = usable.groupby("stock_code").size()
    prod["typical_price"] = median_price
    prod["train_return_rate"] = lines[lines["split"] == "train"].groupby("stock_code")["returned"].mean()
    prod["history_lines"] = prod["n_lines"]
    prod["history_returns"] = cancellations.groupby("stock_code").size()
    prod = prod.fillna({"n_lines_usable": 0, "train_return_rate": np.nan, "history_returns": 0})
    prod["typical_price"] = prod["typical_price"].fillna(lines.groupby("stock_code")["price"].median())
    prod = prod.reset_index().rename(columns={"index": "stock_code"})

    rules = tag_products_with_rules(prod)
    rules.to_csv(cfg.RULE_TAGS_FILE, index=False)
    tags = rules.set_index("stock_code")
    if tag_mode == "llm":
        if not cfg.LLM_TAGS_FILE.exists():
            raise SystemExit(f"--tags llm but {cfg.LLM_TAGS_FILE} does not exist; run fairreturn.genai.product_tagger")
        llm = pd.read_csv(cfg.LLM_TAGS_FILE, dtype={"stock_code": str}, keep_default_na=False).set_index("stock_code")
        if llm["tag_source"].str.contains("mock").any():
            print("  [warn] LLM tags come from the MOCK backend - fine for testing, not for reported results")
        llm = llm[llm["llm_valid"] == 1]
        cols = ["tag_category", "tag_material", "tag_fragile", "tag_seasonal", "tag_source"]
        common = tags.index.intersection(llm.index)
        tags.loc[common, cols] = llm.loc[common, cols].to_numpy()
        print(f"  LLM tags applied to {len(common)} products; rules for the remaining {len(tags) - len(common)}")
    prod = prod.merge(tags[["tag_category", "tag_material", "tag_fragile", "tag_seasonal", "tag_source"]]
                      .reset_index(), on="stock_code", how="left")
    prod["tag_fragile"] = prod["tag_fragile"].astype(int)
    prod["tag_seasonal"] = prod["tag_seasonal"].astype(int)
    return prod


def customer_snapshot(lines: pd.DataFrame, cancellations: pd.DataFrame) -> pd.DataFrame:
    g = lines.groupby("customer_id")
    snap = pd.DataFrame({
        "invoices": g["invoice"].nunique(), "lines": g.size(), "spend": g["line_value"].sum(),
        "first_date": g["invoice_date"].min(), "last_date": g["invoice_date"].max(),
        "country": g["country"].agg(lambda s: s.value_counts().index[0]),
    })
    snap["returns"] = cancellations.groupby("customer_id").size()
    snap["returns"] = snap["returns"].fillna(0).astype(int)
    snap["group"] = snap["country"].map(cfg.country_group)
    return snap.reset_index()


def group_table(lines: pd.DataFrame) -> list[dict]:
    t = (lines[lines["split"].isin(["train", "val", "test"])]
         .groupby(["split", "group"])
         .agg(lines=("returned", "size"), returns=("returned", "sum"), rate=("returned", "mean"),
              customers=("customer_id", "nunique"), invoices=("invoice", "nunique"))
         .reset_index())
    t["rate"] = t["rate"].round(5)
    return t.to_dict(orient="records")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw-dir", default=None)
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--horizon", type=int, default=cfg.HORIZON_DAYS)
    ap.add_argument("--tags", choices=["rules", "llm"], default="rules")
    ap.add_argument("--keep-duplicates", action="store_true")
    a = ap.parse_args()
    t0 = time.time()
    cfg.DATA_PROCESSED.mkdir(parents=True, exist_ok=True)

    print("[1/7] loading raw data")
    tx = loader.load_transactions(a.raw_dir, use_cache=not a.no_cache)
    print("[2/7] cleaning")
    tx, rep_clean = cleaning.clean_transactions(tx, drop_duplicates=not a.keep_duplicates)
    purchases, cancellations, invoices, rep_events = cleaning.split_events(tx)
    data_end = tx["invoice_date"].max()

    print("[3/7] matching returns to purchases")
    lines, rep_labels = labels.label_returns(purchases, cancellations, data_end, a.horizon)
    lines["split"] = assign_splits(lines)
    early = lines["invoice_date"] < pd.Timestamp(cfg.SPLIT_DATES["val_start"])
    median_price = lines.loc[early].groupby("stock_code")["price"].median()

    print("[4/7] point-in-time history features")
    lines = features.add_history(lines, cancellations, invoices, median_price)
    hist = early
    p0 = float(cancellations[cancellations["invoice_date"] < pd.Timestamp(cfg.SPLIT_DATES["val_start"])].shape[0]
               / max(1, int(hist.sum())))

    print("[5/7] product catalogue and tags")
    products = build_products(lines, cancellations, median_price, a.tags)
    products.to_csv(cfg.PRODUCTS_FILE, index=False)
    tagcols = ["tag_category", "tag_material", "tag_fragile", "tag_seasonal"]
    lines = lines.merge(products[["stock_code"] + tagcols + ["tag_source"]], on="stock_code", how="left")

    derived = features.finalize_features(lines, p0)
    for c in derived.columns:
        lines[c] = derived[c].to_numpy()

    print("[6/7] counterfactual twins for the test split")
    inv_train = invoices[invoices["invoice"].isin(lines.loc[lines["split"] == "train", "invoice"])]
    postage_rates = counterfactual.group_postage_rates(inv_train)
    country_weights = counterfactual.group_country_weights(inv_train)
    test = lines[lines["split"] == "test"]
    twins = counterfactual.make_twins(test, postage_rates, country_weights, seed=cfg.SEED)

    print("[7/7] saving")
    keep_ids = ["line_id", "invoice", "stock_code", "description", "customer_id", "country", "invoice_date",
                "split", "returned", "return_lag_days", "returned_qty", "line_value", "tag_source"]
    base = [c for c in features.BASE_COLUMNS if c not in ("invoice_date", "group")]
    model_cols = sorted(set(features.NUMERIC + features.PROXY + features.CATEGORICAL + features.SENSITIVE))
    cols = list(dict.fromkeys(keep_ids + base + model_cols))
    lines[cols].to_pickle(cfg.LINES_FILE)
    twins[cols + ["twin_of", "is_twin"]].to_pickle(cfg.TWINS_FILE)
    customer_snapshot(lines, cancellations).to_csv(cfg.CUSTOMER_SNAPSHOT_FILE, index=False)

    usable = lines[lines["split"].isin(["train", "val", "test"])]
    split_sizes = {s: {"lines": int((lines["split"] == s).sum()),
                       "return_rate": round(float(lines.loc[lines["split"] == s, "returned"].mean()), 5)}
                   for s in ["burnin", "train", "val", "test", "censored"]}
    meta = {
        "created": time.strftime("%Y-%m-%d %H:%M:%S"), "data_end": str(data_end), "horizon_days": a.horizon,
        "split_dates": cfg.SPLIT_DATES, "tags": a.tags, "p0": p0,
        "smoothing_alpha": {"customer": cfg.SMOOTH_ALPHA_CUSTOMER, "product": cfg.SMOOTH_ALPHA_PRODUCT},
        "groups": cfg.GROUPS, "postage_rates": postage_rates, "country_weights": country_weights,
        "report": {**rep_clean, **rep_events, **rep_labels}, "split_sizes": split_sizes,
        "group_table": group_table(lines),
        "usable_return_rate_by_group": usable.groupby("group")["returned"].mean().round(5).to_dict(),
        "features": {"numeric": features.NUMERIC, "proxy": features.PROXY, "categorical": features.CATEGORICAL,
                     "sensitive": features.SENSITIVE, "feature_sets": features.FEATURE_SETS},
        "twins": {"n": int(len(twins)), "seed": cfg.SEED},
    }
    cfg.META_FILE.write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")

    r = meta["report"]
    print(f"\nDone in {time.time() - t0:.0f}s")
    print(f"  raw rows {r['raw_rows']:,} -> duplicates removed {r.get('duplicate_rows_removed', 0):,}; "
          f"without customer {r['rows_without_customer']:,} ({r['share_without_customer']:.1%})")
    print(f"  purchase lines {r['purchase_lines']:,}; cancellation lines {r['cancellation_lines']:,} "
          f"({r['cancellations_matched_share']:.1%} matched to a purchase; median lag {r['lag_days_median']} days)")
    for s, v in split_sizes.items():
        print(f"  {s:9s} {v['lines']:>8,} lines   return rate {v['return_rate']:.2%}")
    print("  return rate by country group (train+val+test):",
          {k: f"{v:.2%}" for k, v in meta["usable_return_rate_by_group"].items()})


if __name__ == "__main__":
    main()
