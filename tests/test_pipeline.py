"""Data pipeline tests on a tiny hand-made transaction table (no real data needed)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from fairreturn.data import cleaning, counterfactual, features, labels


def _tx():
    rows = [
        # invoice, stock, desc, qty, date, price, customer, country
        ("100", "85123A", "WHITE HANGING HEART T-LIGHT HOLDER", 6, "2010-03-01 10:00", 2.55, "1", "United Kingdom"),
        ("100", "22423", "REGENCY CAKESTAND 3 TIER", 2, "2010-03-01 10:00", 12.75, "1", "United Kingdom"),
        ("101", "85123A", "WHITE HANGING HEART T-LIGHT HOLDER", 12, "2010-03-05 11:00", 2.55, "2", "Germany"),
        ("101", "POST", "POSTAGE", 1, "2010-03-05 11:00", 18.0, "2", "Germany"),
        ("C102", "85123A", "WHITE HANGING HEART T-LIGHT HOLDER", -12, "2010-03-20 09:00", 2.55, "2", "Germany"),
        ("103", "22423", "REGENCY CAKESTAND 3 TIER", 1, "2010-04-02 15:00", 12.75, "1", "United Kingdom"),
        ("103", "22423", "REGENCY CAKESTAND 3 TIER", 1, "2010-04-02 15:00", 12.75, "1", "United Kingdom"),  # exact duplicate
        ("104", "21212", "PACK OF 72 RETROSPOT CAKE CASES", 24, "2010-04-03 12:00", 0.55, "", "United Kingdom"),  # no customer
        ("C105", "22423", "REGENCY CAKESTAND 3 TIER", -1, "2010-08-30 12:00", 12.75, "1", "United Kingdom"),  # > 60 days later
        ("106", "85123A", "WHITE HANGING HEART T-LIGHT HOLDER", 6, "2010-05-01 10:00", 2.55, "1", "United Kingdom"),
    ]
    df = pd.DataFrame(rows, columns=["invoice", "stock_code", "description", "quantity", "invoice_date", "price",
                                     "customer_id", "country"])
    df["invoice_date"] = pd.to_datetime(df["invoice_date"])
    return df


def _pipeline():
    tx, rep = cleaning.clean_transactions(_tx())
    purchases, canc, inv, rep2 = cleaning.split_events(tx)
    data_end = pd.Timestamp("2011-01-01")
    lines, rep3 = labels.label_returns(purchases, canc, data_end, horizon_days=60)
    med = lines.groupby("stock_code")["price"].median()
    lines = features.add_history(lines, canc, inv, med)
    return tx, purchases, canc, inv, lines, {**rep, **rep2, **rep3}


def test_cleaning_counts():
    tx, purchases, canc, inv, lines, rep = _pipeline()
    assert rep["duplicate_rows_removed"] == 1
    assert rep["rows_without_customer"] == 1
    assert len(purchases) == 5          # POST line is a service code, not a purchase
    assert len(canc) == 2
    assert inv.set_index("invoice").loc["101", "has_postage"] == 1
    assert inv.set_index("invoice").loc["100", "has_postage"] == 0


def test_return_labels_and_horizon():
    *_, lines, rep = _pipeline()
    by = lines.set_index(["invoice", "stock_code"])
    assert by.loc[("101", "85123A"), "returned"] == 1                    # returned after 15 days
    assert abs(by.loc[("101", "85123A"), "return_lag_days"] - 14.92) < 0.1
    assert by.loc[("103", "22423"), "returned"] == 0                     # C105 is > 60 days after 103
    assert by.loc[("100", "22423"), "returned"] == 0                     # C105 matches the LATEST earlier purchase (103)
    assert lines["returned"].sum() == 1


def test_history_is_point_in_time():
    *_, lines, _ = _pipeline()
    r = lines.set_index(["invoice", "stock_code"])
    first = r.loc[("100", "85123A")]
    assert first["cust_prior_lines"] == 0 and first["cust_prior_invoices"] == 0     # first ever order
    later = r.loc[("106", "85123A")]
    assert later["cust_prior_lines"] == 3 and later["cust_prior_invoices"] == 2     # 100 (2 lines) + 103 (1 line)
    assert later["prod_prior_lines"] == 2 and later["prod_prior_returns"] == 1      # sold in 100 and 101; returned C102
    g = r.loc[("101", "85123A")]
    assert g["cust_prior_returns"] == 0          # the return happens AFTER the purchase - must not leak
    assert g["has_postage"] == 1 and g["basket_lines"] == 1


def test_finalize_and_builder_roundtrip(tmp_path=None):
    *_, lines, _ = _pipeline()
    lines["tag_category"], lines["tag_material"], lines["tag_fragile"], lines["tag_seasonal"] = "home_decor", "glass", 1, 0
    d = features.finalize_features(lines, p0=0.02)
    assert d["cust_prior_return_rate"].between(0, 1).all()
    for fs in features.FEATURE_SETS:
        fb = features.FeatureBuilder(fs).fit(d)
        X = fb.transform(d)
        assert X.shape == (len(d), fb.n_features) and np.isfinite(X).all()
        fb2 = features.FeatureBuilder.from_dict(fb.to_dict())
        assert np.allclose(fb2.transform(d), X)
    aware = features.FeatureBuilder("aware").fit(d)
    assert any(n.startswith("group=") for n in aware.feature_names)
    unaware = features.FeatureBuilder("unaware").fit(d)
    assert not any(n.startswith("group=") for n in unaware.feature_names) and "has_postage" in unaware.feature_names
    blind = features.FeatureBuilder("blind").fit(d)
    assert "has_postage" not in blind.feature_names


def test_twins_change_only_country_and_postage():
    *_, inv, lines, _ = _pipeline()[2:]
    lines = lines.copy()
    lines["tag_category"], lines["tag_material"], lines["tag_fragile"], lines["tag_seasonal"] = "home_decor", "glass", 1, 0
    rates = counterfactual.group_postage_rates(inv)
    weights = counterfactual.group_country_weights(inv)
    tw = counterfactual.make_twins(lines, rates, weights, seed=3)
    assert (tw["group"].to_numpy() != lines["group"].to_numpy()).all()
    keep = ["quantity", "price", "cust_prior_lines", "prod_prior_lines", "returned", "basket_value"]
    assert (tw[keep].to_numpy() == lines[keep].to_numpy()).all()
    for _, g in tw.groupby("invoice"):
        assert g["group"].nunique() == 1          # a whole invoice moves together
