"""Service and REST API tests. They need the trained outputs (run `python run_pipeline.py` first) and are
skipped otherwise."""
from __future__ import annotations

import json

from fairreturn import config as cfg

ORDER = {"stock_code": "85123A", "quantity": 6, "unit_price": 2.55, "basket_lines": 20, "basket_value": 300,
         "order_date": "2011-11-15T11:30", "country": "Germany", "has_postage": "auto",
         "customer": {"prior_invoices": 8, "prior_items": 190, "prior_returns": 2, "tenure_days": 300,
                      "recency_days": 30}}


def test_frontend_assets_are_packaged():
    from fairreturn.backend.app import create_app
    app = create_app(service=object())
    app.testing = True
    c = app.test_client()
    index = c.get("/")
    assert index.status_code == 200 and b"Fair Returns" in index.data
    assert c.get("/static/css/app.css").status_code == 200
    assert c.get("/static/js/app.js").status_code == 200


def _client():
    if not (cfg.MODELS_DIR / cfg.DEMO_OURS).exists() or not (cfg.DEMO_DIR / "invoices.csv").exists():
        return None
    from fairreturn.backend.app import create_app
    from fairreturn.backend.service import FairReturnService
    app = create_app(FairReturnService(llm_backend="mock"))
    app.testing = True
    return app.test_client()


def test_api_endpoints():
    c = _client()
    if c is None:
        return "skipped (run the pipeline first)"
    assert c.get("/api/health").get_json()["status"] == "ok"
    meta = c.get("/api/meta").get_json()
    assert meta["groups"] == cfg.GROUPS and meta["models"]
    prods = c.get("/api/products?q=heart&limit=5").get_json()
    assert prods and all("HEART" in p["description"] for p in prods)

    pred = c.post("/api/predict", json=ORDER).get_json()
    names = {m["name"] for m in pred["models"]}
    assert cfg.DEMO_OURS in names and cfg.DEMO_BASELINE in names
    for m in pred["models"]:
        assert 0 < m["risk"] < 1 and m["factors"]
    ours = next(m for m in pred["models"] if m["name"] == cfg.DEMO_OURS)
    groups_used = {f["group"] for f in ours["factors"]}
    assert "Country" not in groups_used and "Postage (country proxy)" not in groups_used  # never inputs of ours

    cf = c.post("/api/counterfactual", json={**ORDER, "proxy_mode": "with_proxies"}).get_json()
    assert len(cf["models"][cfg.DEMO_OURS]["risks"]) == len(cfg.GROUPS)
    # moving the customer (and the postage line with them) cannot change our model's answer at all
    assert cf["models"][cfg.DEMO_OURS]["spread"] < 1e-9
    cf0 = c.post("/api/counterfactual", json={**ORDER, "proxy_mode": "country_only"}).get_json()
    # without moving the postage line, a model that has no country column cannot change either
    assert cf0["models"][cfg.DEMO_OURS]["spread"] < 1e-9
    if "gbdt_aware" in cf0["models"]:
        assert cf0["models"]["gbdt_aware"]["spread"] > 0          # ... while the country-aware baseline does

    a = c.post("/api/assistant", json=ORDER).get_json()
    assert a["action"] and a["message"] and a["source"] == "rules"

    inv = c.get("/api/invoices?limit=3").get_json()
    assert inv["total"] > 0
    ranked = c.get("/api/invoices?sort=country_effect&limit=20").get_json()["invoices"]
    effects = [i["country_effect"] for i in ranked]
    assert effects == sorted(effects, reverse=True)
    first = inv["invoices"][0]["invoice"]
    d = c.get(f"/api/invoices/{first}").get_json()
    assert d["lines"] and d["n_lines"] == len(d["lines"])
    other = next(g for g in cfg.GROUPS if g != d["original_group"])
    d2 = c.get(f"/api/invoices/{first}?as_group={other}").get_json()
    assert d2["shown_as_group"] == other

    assert c.get("/api/results").status_code == 200
    assert c.get("/api/fairness").status_code == 200
    assert c.get("/api/tags").status_code == 200
    assert c.get("/api/audit").status_code == 200
    assert c.get("/api/examples").status_code == 200
    assert c.get("/api/nope").status_code == 404
    bad = c.post("/api/predict", json={**ORDER, "unit_price": -1})
    assert bad.status_code == 400 and "error" in bad.get_json()
    assert c.get("/").status_code == 200 and b"Fair Returns" in c.get("/").data


def test_score_csv():
    c = _client()
    if c is None:
        return "skipped (run the pipeline first)"
    csv = ("Invoice,StockCode,Description,Quantity,InvoiceDate,Price,Customer ID,Country\n"
           "600001,85123A,WHITE HANGING HEART T-LIGHT HOLDER,6,2011-12-12 10:00,2.55,12347,Iceland\n"
           "600001,22423,REGENCY CAKESTAND 3 TIER,2,2011-12-12 10:00,12.75,12347,Iceland\n"
           "600001,POST,POSTAGE,1,2011-12-12 10:00,40,12347,Iceland\n"
           "600002,99999X,NEW GLASS VASE,4,2011-12-12 11:00,3.50,,United Kingdom\n")
    r = c.post("/api/score-csv", data=csv, content_type="text/csv")
    body = r.get_json()
    assert r.status_code == 200, body
    assert body["rows"] == 3 and f"risk_{cfg.DEMO_OURS}" in body["csv"]
    r2 = c.post("/api/score-csv?format=csv", data=csv, content_type="text/csv")
    assert r2.status_code == 200 and r2.mimetype == "text/csv"
    json.dumps(body)
