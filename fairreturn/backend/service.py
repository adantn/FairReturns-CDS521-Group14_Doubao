"""Framework-independent service layer behind the REST API.

Everything the web app needs - predictions, counterfactuals, real invoices, results, the LLM audit and the
decision assistant - is implemented here as plain Python returning JSON-ready dicts, so it can be tested
without a web server and reused from notebooks:

    from fairreturn.backend.service import FairReturnService
    svc = FairReturnService()
    svc.predict({"stock_code": "22423", "quantity": 4, "country": "Germany"})
"""
from __future__ import annotations

import io
import json
import math
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from fairreturn import config as cfg
from fairreturn.data import cleaning, loader
from fairreturn.data.features import finalize_features
from fairreturn.evaluation import explain
from fairreturn.genai import assistant, prompts
from fairreturn.genai.llm_client import LLMClient, api_configured
from fairreturn.genai.product_tagger import rule_tag
from fairreturn.models.registry import LoadedModel

DEFAULT_ORDER_DATE = "2011-11-15 11:30"
MAX_UPLOAD_ROWS = 20000


class ApiError(Exception):
    def __init__(self, message: str, status: int = 400):
        super().__init__(message)
        self.message, self.status = message, status


def jsonable(obj):
    """numpy / pandas / NaN -> plain JSON types."""
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        v = float(obj)
        return None if (math.isnan(v) or math.isinf(v)) else v
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (pd.Timestamp, datetime)):
        return obj.isoformat()
    if isinstance(obj, np.ndarray):
        return jsonable(obj.tolist())
    if obj is pd.NA or obj is pd.NaT:
        return None
    return obj


def _read_json(path: Path, default=None):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default


class FairReturnService:
    def __init__(self, outputs: Path | None = None, llm_backend: str = "auto"):
        self.outputs = Path(outputs or cfg.OUTPUTS)
        self.meta = _read_json(cfg.META_FILE)
        if self.meta is None:
            raise FileNotFoundError("data/processed/meta.json missing - run `python run_pipeline.py` first")
        self.p0 = float(self.meta["p0"])
        self.postage_rates = self.meta["postage_rates"]
        self.country_weights = self.meta["country_weights"]
        self.representative = {g: max(w, key=w.get) for g, w in self.country_weights.items()}
        self.products = pd.read_csv(cfg.PRODUCTS_FILE, dtype={"stock_code": str}, keep_default_na=False,
                                    na_values=[""]).set_index("stock_code")
        self.customers = (pd.read_csv(cfg.CUSTOMER_SNAPSHOT_FILE, dtype={"customer_id": str},
                                      parse_dates=["first_date", "last_date"]).set_index("customer_id")
                          if cfg.CUSTOMER_SNAPSHOT_FILE.exists() else pd.DataFrame())
        self.models: dict[str, LoadedModel] = {}
        for name in cfg.DEMO_MODELS:
            try:
                self.models[name] = LoadedModel(name, self.outputs / "models")
            except FileNotFoundError:
                continue
        if not self.models:
            raise FileNotFoundError("no trained models in outputs/models - run `python run_pipeline.py` first")
        self.baseline = cfg.DEMO_BASELINE if cfg.DEMO_BASELINE in self.models else next(iter(self.models))
        self.ours = cfg.DEMO_OURS if cfg.DEMO_OURS in self.models else list(self.models)[-1]
        self.thresholds = {n: float(m.info["metrics"]["threshold"]) for n, m in self.models.items()}
        demo = self.outputs / "demo"
        self.demo_lines = pd.read_pickle(demo / "invoice_lines.pkl") if (demo / "invoice_lines.pkl").exists() else None
        self.demo_invoices = (pd.read_csv(demo / "invoices.csv", dtype={"invoice": str, "customer_id": str},
                                          parse_dates=["date"]) if (demo / "invoices.csv").exists() else None)
        self.llm_backend = llm_backend
        self._llm = None
        self._examples = None
        self._swings = None

    # ------------------------------------------------------------------ helpers
    def _llm_client(self) -> LLMClient | None:
        if self._llm is None:
            backend = self.llm_backend
            if backend == "auto":
                backend = "api" if api_configured() else "mock"
            try:
                self._llm = LLMClient(backend=backend, cache_file=self.outputs / "genai" / "llm_cache_assistant.jsonl")
            except Exception:  # noqa: BLE001
                self._llm = LLMClient(backend="mock")
        return self._llm

    def _resolve_country(self, country: str | None) -> tuple[str, str]:
        c = (country or "United Kingdom").strip()
        if c in cfg.GROUPS and c not in ("United Kingdom", "Germany", "France", "EIRE"):
            return self.representative.get(c, c), c
        return c, cfg.country_group(c)

    def _postage_for(self, group: str) -> int:
        return int(self.postage_rates.get(group, 0.0) >= 0.5)

    def base_from_order(self, o: dict) -> pd.DataFrame:
        """Turn the order form (or an API payload) into one row of base columns."""
        sc = str(o.get("stock_code") or "").strip().upper()
        prod = self.products.loc[sc] if sc in self.products.index else None
        desc = str(o.get("description") or (prod["description"] if prod is not None else "UNKNOWN PRODUCT")).upper()
        if prod is not None:
            tags = {"category": prod["tag_category"], "material": prod["tag_material"],
                    "fragile": int(prod["tag_fragile"]), "seasonal": int(prod["tag_seasonal"])}
        else:
            tags = rule_tag(desc)
        try:
            qty = max(1, int(float(o.get("quantity", 1))))
            price = float(o.get("unit_price") or (prod["typical_price"] if prod is not None else 1.0))
            basket_lines = max(1, int(float(o.get("basket_lines", 1))))
        except (TypeError, ValueError) as e:
            raise ApiError(f"invalid number in order: {e}") from e
        if price <= 0:
            raise ApiError("unit_price must be positive")
        line_value = qty * price
        basket_value = max(float(o.get("basket_value") or 0), line_value)
        basket_qty = max(float(o.get("basket_qty") or qty * basket_lines), qty)
        when = pd.Timestamp(o.get("order_date") or DEFAULT_ORDER_DATE)
        country, group = self._resolve_country(o.get("country"))
        hp = o.get("has_postage", "auto")
        has_postage = self._postage_for(group) if hp in ("auto", None, "") else int(bool(hp) and str(hp) != "false")

        cust = o.get("customer") or {}
        cid = str(cust.get("customer_id") or "").strip()
        if cid and cid in self.customers.index:
            s = self.customers.loc[cid]
            c = {"cust_prior_invoices": s["invoices"], "cust_prior_lines": s["lines"], "cust_prior_returns": s["returns"],
                 "cust_prior_spend": s["spend"], "cust_tenure_days": max(0, (when - s["first_date"]).days),
                 "cust_recency_days": max(1, (when - s["last_date"]).days)}
        elif cust.get("new", False) or not cust:
            c = {"cust_prior_invoices": 0, "cust_prior_lines": 0, "cust_prior_returns": 0, "cust_prior_spend": 0.0,
                 "cust_tenure_days": 0, "cust_recency_days": 365}
        else:
            items = float(cust.get("prior_items", 0))
            c = {"cust_prior_invoices": float(cust.get("prior_invoices", 0)), "cust_prior_lines": items,
                 "cust_prior_returns": float(cust.get("prior_returns", 0)),
                 "cust_prior_spend": float(cust.get("prior_spend") or items * 17.0),
                 "cust_tenure_days": float(cust.get("tenure_days", 180)),
                 "cust_recency_days": float(cust.get("recency_days", 30))}
            if c["cust_prior_returns"] > max(c["cust_prior_lines"], 0) + 1000:
                raise ApiError("prior_returns looks implausible")
        row = {
            "stock_code": sc or "CUSTOM", "description": desc, "quantity": qty, "price": price,
            "product_median_price": float(prod["typical_price"]) if prod is not None else np.nan,
            "basket_lines": basket_lines, "basket_value": basket_value, "basket_qty": basket_qty,
            "has_postage": has_postage, **c,
            "prod_prior_lines": float(prod["history_lines"]) if prod is not None else 0.0,
            "prod_prior_returns": float(prod["history_returns"]) if prod is not None else 0.0,
            "invoice_date": when, "tag_category": tags["category"], "tag_material": tags["material"],
            "tag_fragile": int(tags["fragile"]), "tag_seasonal": int(tags["seasonal"]),
            "group": group, "country": country,
        }
        return pd.DataFrame([row])

    def _score(self, base: pd.DataFrame, names: list[str] | None = None, with_factors: bool = True) -> dict:
        derived = finalize_features(base, self.p0)
        out = {}
        for n in names or list(self.models):
            lm = self.models[n]
            X = lm.matrix(derived)
            p = lm.predict_matrix(X)
            res = {"risk": p}
            if with_factors:
                res["factors"] = explain.contributions(lm.predict_matrix, X, lm.fb, lm.reference)
            out[n] = res
        return out

    def _model_meta(self, n: str) -> dict:
        lm = self.models[n]
        m = lm.info.get("metrics", {})
        return {"name": n, "label": lm.label, "role": lm.info.get("role"), "feature_set": lm.info.get("feature_set"),
                "threshold": self.thresholds[n], "auc": m.get("auc"), "fpr_gap": m.get("fpr_gap"),
                "tpr_gap": m.get("tpr_gap"), "cf_flip_rate": m.get("cf_flip_rate"),
                "params": m.get("params"), "size_kb": m.get("size_kb"), "uses_country": lm.info.get("feature_set") == "aware"}

    # ------------------------------------------------------------------ endpoints
    def health(self) -> dict:
        return {"status": "ok", "models": list(self.models), "baseline": self.baseline, "ours": self.ours,
                "demo_invoices": 0 if self.demo_invoices is None else int(len(self.demo_invoices)),
                "llm": self._llm_client().describe(), "data_created": self.meta.get("created")}

    def meta_info(self) -> dict:
        r = self.meta["report"]
        return jsonable({
            "groups": cfg.GROUPS, "group_short": cfg.GROUP_SHORT, "representative_country": self.representative,
            "postage_rates": self.postage_rates, "countries": sorted({c for w in self.country_weights.values() for c in w}),
            "countries_by_group": self.country_weights,
            "models": [self._model_meta(n) for n in self.models], "baseline": self.baseline, "ours": self.ours,
            "high_risk_share": cfg.HIGH_RISK_SHARE, "horizon_days": self.meta["horizon_days"],
            "split_dates": self.meta["split_dates"], "split_sizes": self.meta["split_sizes"], "report": r,
            "group_table": self.meta["group_table"], "tags": self.meta.get("tags"),
            "return_rate_by_group": self.meta.get("usable_return_rate_by_group"),
            "llm": self._llm_client().describe(),
        })

    def search_products(self, q: str = "", limit: int = 30) -> list[dict]:
        p = self.products
        if q:
            qq = q.strip().upper()
            p = p[p["description"].str.contains(qq, regex=False, na=False) | (p.index == qq)]
        p = p.sort_values("n_lines", ascending=False).head(max(1, min(limit, 200)))
        return jsonable([{"stock_code": sc, "description": r["description"], "typical_price": r["typical_price"],
                          "n_lines": r["n_lines"], "history_returns": r["history_returns"],
                          "tag_category": r["tag_category"], "tag_material": r["tag_material"],
                          "tag_fragile": r["tag_fragile"], "tag_seasonal": r["tag_seasonal"],
                          "tag_source": r.get("tag_source")} for sc, r in p.iterrows()])

    def predict(self, order: dict) -> dict:
        base = self.base_from_order(order)
        scores = self._score(base)
        row = base.iloc[0]
        models = []
        for n, s in scores.items():
            f = explain.top_factors(s["factors"][0], k=7, min_abs=0)
            models.append({**self._model_meta(n), "risk": float(s["risk"][0]),
                           "high_risk": bool(s["risk"][0] >= self.thresholds[n]),
                           "factors": [{"group": g, "delta": d} for g, d in f]})
        return jsonable({"order": {k: row[k] for k in base.columns}, "facts": prompts.order_facts(row),
                         "models": models, "baseline": self.baseline, "ours": self.ours})

    def counterfactual(self, order: dict, proxy_mode: str = "with_proxies") -> dict:
        base = self.base_from_order(order)
        given_country, given_group = base.loc[0, "country"], base.loc[0, "group"]
        rows = []
        for g in cfg.GROUPS:
            r = base.iloc[0].copy()
            r["group"] = g
            r["country"] = given_country if g == given_group else self.representative.get(g, g)
            if proxy_mode == "with_proxies":
                r["has_postage"] = self._postage_for(g)
            rows.append(r)
        cf = pd.DataFrame(rows).reset_index(drop=True)
        scores = self._score(cf, with_factors=False)
        out = {"groups": cfg.GROUPS, "countries": cf["country"].tolist(), "postage": cf["has_postage"].tolist(),
               "proxy_mode": proxy_mode, "original_group": given_group, "models": {}}
        for n, s in scores.items():
            r = s["risk"]
            out["models"][n] = {"label": self.models[n].label, "risks": r.tolist(),
                                "flags": (r >= self.thresholds[n]).tolist(), "threshold": self.thresholds[n],
                                "spread": float(r.max() - r.min()), "ratio": float(r.max() / max(r.min(), 1e-9))}
        return jsonable(out)

    def assistant(self, order: dict) -> dict:
        base = self.base_from_order(order)
        s = self._score(base, [self.ours])[self.ours]
        risk = float(s["risk"][0])
        factors = explain.top_factors(s["factors"][0], k=5)
        ans = assistant.assist(base.iloc[0], risk, self.thresholds[self.ours], factors, self._llm_client())
        return jsonable({"model": self.ours, "risk": risk, "threshold": self.thresholds[self.ours], **ans})

    def examples(self) -> list[dict]:
        """Preset orders for the demo: real test lines where the country-aware baseline swings most."""
        if self._examples is not None:
            return self._examples
        ex = []
        if self.demo_lines is not None and self.baseline in self.models:
            d = self.demo_lines
            cand = pd.concat([d[d["group"] != "United Kingdom"].sample(min(150, (d["group"] != "United Kingdom").sum()), random_state=1),
                              d[d["group"] == "United Kingdom"].sample(min(150, (d["group"] == "United Kingdom").sum()), random_state=1)])
            cand = cand.reset_index(drop=True)
            # every candidate in every country group, scored in one batch: shape (groups, candidates)
            moved = pd.concat([cand.assign(group=g, has_postage=float(self._postage_for(g))) for g in cfg.GROUPS],
                              ignore_index=True)
            sc = self._score(moved, [self.baseline], with_factors=False)[self.baseline]["risk"]
            sc = np.asarray(sc).reshape(len(cfg.GROUPS), len(cand))
            cand = cand.assign(spread=sc.max(axis=0) - sc.min(axis=0)).sort_values("spread", ascending=False)
            for _, r in pd.concat([cand.head(4), cand.tail(1)]).iterrows():
                # the catalogue description, which is what the product search shows (a code can have several)
                desc = self.products.loc[r["stock_code"], "description"] if r["stock_code"] in self.products.index \
                    else r["description"]
                ex.append(self._order_from_line(r, title=f"{str(desc).title()} · {r['country']}"))
        self._examples = jsonable(ex)
        return self._examples

    def _order_from_line(self, r, title: str = "") -> dict:
        new = float(r["cust_prior_invoices"]) == 0
        return {"title": title, "stock_code": r["stock_code"], "description": r["description"],
                "quantity": int(r["quantity"]), "unit_price": float(r["price"]), "basket_lines": int(r["basket_lines"]),
                "basket_value": float(r["basket_value"]), "order_date": pd.Timestamp(r["invoice_date"]).isoformat(),
                "country": r["country"], "has_postage": int(r["has_postage"]),
                "customer": {"new": True} if new else {
                    "prior_invoices": int(r["cust_prior_invoices"]), "prior_items": int(r["cust_prior_lines"]),
                    "prior_returns": int(r["cust_prior_returns"]), "prior_spend": float(r["cust_prior_spend"]),
                    "tenure_days": int(r["cust_tenure_days"]), "recency_days": int(r["cust_recency_days"])},
                "actual_returned": int(r.get("returned", 0))}

    # ------------------------------------------------------------------ real invoices
    def invoice_swings(self) -> pd.Series:
        """Per demo invoice: how many more lines the country-aware baseline flags than it would if the customer
        lived in the UK (for UK invoices: in Germany, sign flipped). Computed once, in two batch predictions."""
        if self._swings is None:
            d = self.demo_lines.reset_index(drop=True)
            n = self.baseline
            uk = (d["group"] == "United Kingdom").to_numpy()
            alt = np.where(uk, "Germany", "United Kingdom")
            moved = d.assign(group=alt, country=[self.representative.get(g, g) for g in alt],
                             has_postage=[float(self._postage_for(g)) for g in alt])
            f0 = self._score(d, [n], with_factors=False)[n]["risk"] >= self.thresholds[n]
            f1 = self._score(moved, [n], with_factors=False)[n]["risk"] >= self.thresholds[n]
            delta = np.where(uk, f1.astype(int) - f0.astype(int), f0.astype(int) - f1.astype(int))
            self._swings = pd.Series(delta, index=d.index).groupby(d["invoice"].to_numpy()).sum()
        return self._swings

    def list_invoices(self, group: str | None = None, has_return: str | None = None, q: str | None = None,
                      limit: int = 50, offset: int = 0, sort: str | None = None) -> dict:
        """sort='country_effect': invoices where the baseline's flags depend most on the country come first."""
        if self.demo_invoices is None:
            raise ApiError("demo bundle missing - run `python -m fairreturn.backend.demo_bundle`", 404)
        d = self.demo_invoices
        if self.demo_lines is not None and self.baseline in self.models:
            d = d.assign(country_effect=d["invoice"].map(self.invoice_swings()).fillna(0).astype(int))
        if sort == "country_effect" and "country_effect" in d:
            d = d.sort_values(["country_effect", "date"], ascending=[False, True])
        if group:
            d = d[d["group"] == group]
        if has_return in ("1", "0"):
            d = d[d["has_return"] == int(has_return)]
        if q:
            d = d[d["invoice"].str.contains(q.strip(), regex=False) | d["customer_id"].astype(str).str.contains(q.strip(), regex=False)]
        total = int(len(d))
        d = d.iloc[offset: offset + max(1, min(limit, 200))]
        return jsonable({"total": total, "invoices": d.to_dict(orient="records")})

    def invoice_detail(self, invoice: str, as_group: str | None = None) -> dict:
        if self.demo_lines is None:
            raise ApiError("demo bundle missing", 404)
        lines = self.demo_lines[self.demo_lines["invoice"] == str(invoice)].copy()
        if lines.empty:
            raise ApiError(f"invoice {invoice} is not in the demo sample", 404)
        orig_group, orig_country = lines["group"].iloc[0], lines["country"].iloc[0]
        if as_group and as_group != orig_group:
            if as_group not in cfg.GROUPS:
                raise ApiError(f"unknown group {as_group}")
            lines["group"] = as_group
            lines["country"] = self.representative.get(as_group, as_group)
            lines["has_postage"] = float(self._postage_for(as_group))
        scores = self._score(lines.reset_index(drop=True), with_factors=True)
        out_lines = []
        for i, (_, r) in enumerate(lines.reset_index(drop=True).iterrows()):
            item = {"stock_code": r["stock_code"], "description": r["description"], "quantity": int(r["quantity"]),
                    "price": float(r["price"]), "returned": int(r["returned"]),
                    "tag_category": r["tag_category"], "tag_fragile": int(r["tag_fragile"]), "risks": {}, "flags": {}}
            for n, s in scores.items():
                item["risks"][n] = float(s["risk"][i])
                item["flags"][n] = bool(s["risk"][i] >= self.thresholds[n])
            top = explain.top_factors(scores[self.ours]["factors"][i], k=2)
            item["ours_top_factors"] = [{"group": g, "delta": d_} for g, d_ in top]
            out_lines.append(item)
        summary = {n: {"mean_risk": float(np.mean(s["risk"])), "flagged": int(np.sum(s["risk"] >= self.thresholds[n])),
                       "label": self.models[n].label} for n, s in scores.items()}
        first = lines.iloc[0]
        return jsonable({
            "invoice": str(invoice), "date": first["invoice_date"], "customer_id": first["customer_id"],
            "original_group": orig_group, "original_country": orig_country, "shown_as_group": lines["group"].iloc[0],
            "shown_as_country": lines["country"].iloc[0], "has_postage": int(first["has_postage"]),
            "returned_lines": int(lines["returned"].sum()), "n_lines": int(len(lines)),
            "customer": {"prior_invoices": first["cust_prior_invoices"], "prior_items": first["cust_prior_lines"],
                         "prior_returns": first["cust_prior_returns"], "tenure_days": first["cust_tenure_days"]},
            "summary": summary, "lines": out_lines})

    # ------------------------------------------------------------------ results and GenAI
    def results(self) -> dict:
        summ = _read_json(cfg.RESULTS_DIR / "summary.json", [])
        figs = sorted(p.name for p in cfg.FIG_DIR.glob("*.png")) if cfg.FIG_DIR.exists() else []
        sweep = cfg.RESULTS_DIR / "sweep" / "sweep.csv"
        stress = sorted(cfg.RESULTS_DIR.glob("stress_bias_*/summary.json"))
        return jsonable({
            "summary": summ, "figures": figs, "input_probe": _read_json(cfg.RESULTS_DIR / "input_probe.json", {}),
            "run_info": _read_json(cfg.RESULTS_DIR / "run_info.json", {}),
            "sweep": pd.read_csv(sweep).to_dict(orient="records") if sweep.exists() else [],
            "stress": {p.parent.name: _read_json(p, []) for p in stress},
            "summary_md": (cfg.RESULTS_DIR / "summary.md").read_text(encoding="utf-8")
            if (cfg.RESULTS_DIR / "summary.md").exists() else "",
        })

    def fairness(self, method: str | None = None) -> dict:
        f = cfg.RESULTS_DIR / "group_metrics.csv"
        if not f.exists():
            raise ApiError("results missing - run experiments.run_all", 404)
        g = pd.read_csv(f)
        methods = g[["method", "label"]].drop_duplicates().to_dict(orient="records")
        if method:
            g = g[g["method"] == method]
        return jsonable({"methods": methods, "rows": g.to_dict(orient="records")})

    def audit(self) -> dict:
        s = cfg.AUDIT_DIR / "summary.csv"
        p = _read_json(cfg.AUDIT_DIR / "pairs.json", {})
        if not s.exists():
            return {"available": False}
        pairs = p.get("pairs", {})
        brief = [{"pair_id": k, "returned": v["returned"], "original": v["original"], "twin": v["twin"],
                  "plain_delta": (None if v["conditions"]["plain"]["twin"]["risk"] is None
                                  or v["conditions"]["plain"]["original"]["risk"] is None else
                                  v["conditions"]["plain"]["twin"]["risk"] - v["conditions"]["plain"]["original"]["risk"])}
                 for k, v in pairs.items()]
        return jsonable({"available": True, "llm": p.get("llm", {}), "summary": pd.read_csv(s).to_dict(orient="records"),
                         "pairs": brief})

    def audit_pair(self, pair_id: str) -> dict:
        p = _read_json(cfg.AUDIT_DIR / "pairs.json", {}).get("pairs", {})
        if pair_id not in p:
            raise ApiError(f"pair {pair_id} not found", 404)
        return jsonable(p[pair_id])

    def tags(self) -> dict:
        ev = _read_json(cfg.GENAI_DIR / "tag_eval.json")
        rates = None
        if cfg.PRODUCTS_FILE.exists():
            pr = self.products
            rates = {"by_category": pr.groupby("tag_category")["train_return_rate"].mean().round(4).to_dict(),
                     "by_fragile": pr.groupby("tag_fragile")["train_return_rate"].mean().round(4).to_dict()}
        return jsonable({"available": ev is not None, "eval": ev or {}, "product_rates": rates,
                         "tags_used_in_models": self.meta.get("tags")})

    # ------------------------------------------------------------------ batch scoring
    def score_csv(self, text: str) -> tuple[dict, str]:
        """Score raw Online-Retail-format rows (Invoice, StockCode, Description, Quantity, InvoiceDate, Price,
        Customer ID, Country). History comes from the snapshot at the end of the data."""
        try:
            raw = pd.read_csv(io.StringIO(text), dtype=str)
        except Exception as e:  # noqa: BLE001
            raise ApiError(f"could not read CSV: {e}") from e
        if len(raw) > MAX_UPLOAD_ROWS:
            raise ApiError(f"too many rows ({len(raw)}); limit is {MAX_UPLOAD_ROWS}")
        try:
            tx = loader.normalise(loader._rename(raw, "upload"))
        except ValueError as e:
            raise ApiError(str(e)) from e
        tx = cleaning.flag_rows(tx)
        product = ~tx["is_service"] & ~tx["is_adjust"] & ~tx["is_cancel"] & (tx["quantity"] > 0) & (tx["price"] > 0)
        lines = tx[product].copy()
        if lines.empty:
            raise ApiError("no purchase lines found (need positive Quantity and Price, product stock codes)")
        lines["line_value"] = lines["quantity"] * lines["price"]
        inv = lines.groupby("invoice").agg(basket_lines=("stock_code", "size"), basket_value=("line_value", "sum"),
                                           basket_qty=("quantity", "sum"))
        post = set(tx.loc[tx["is_postage"], "invoice"])
        rows = []
        for _, r in lines.iterrows():
            o = {"stock_code": r["stock_code"], "description": r["description"], "quantity": r["quantity"],
                 "unit_price": r["price"], "basket_lines": inv.loc[r["invoice"], "basket_lines"],
                 "basket_value": inv.loc[r["invoice"], "basket_value"], "basket_qty": inv.loc[r["invoice"], "basket_qty"],
                 "order_date": r["invoice_date"], "country": r["country"],
                 "has_postage": int(r["invoice"] in post),
                 "customer": {"customer_id": r["customer_id"]} if r["customer_id"] else {"new": True}}
            if o["customer"].get("customer_id") and o["customer"]["customer_id"] not in self.customers.index:
                o["customer"] = {"new": True}
            rows.append(self.base_from_order(o))
        base = pd.concat(rows, ignore_index=True)
        scores = self._score(base, with_factors=False)
        out = lines[["invoice", "stock_code", "description", "quantity", "price", "customer_id", "country"]].reset_index(drop=True)
        out["group"] = base["group"]
        for n, s in scores.items():
            out[f"risk_{n}"] = np.round(s["risk"], 5)
            out[f"high_risk_{n}"] = (s["risk"] >= self.thresholds[n]).astype(int)
        csv_text = out.to_csv(index=False)
        summary = {n: {"mean_risk": float(np.mean(s["risk"])), "flagged": int(np.sum(s["risk"] >= self.thresholds[n]))}
                   for n, s in scores.items()}
        return jsonable({"rows": int(len(out)), "skipped_rows": int(len(tx) - len(out)), "summary": summary,
                         "preview": out.head(50).to_dict(orient="records")}), csv_text
