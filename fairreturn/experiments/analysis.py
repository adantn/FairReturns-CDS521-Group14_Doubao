"""Extra analyses from the saved models and predictions (no retraining):

  1. Value of the GenAI product tags: AUC with vs without tags, overall and for products with little history
     (cold start), from the GBDT-Blind and GBDT-Blind-without-tags predictions.
  2. Per-group false-alarm, hit-rate and calibration tables for the headline models.
  3. The postage proxy: how much each model that sees the postage line changes its risk when only that line is
     added. Models without an adversary *raise* the risk; VIB + GRL with the proxy kept learns to *lower* it,
     i.e. it uses the proxy in reverse to equalise error rates - which is why our model does not get the
     postage line at all.
  4. Plain-language headline sentences for the report and the slides.

    python -m fairreturn.experiments.analysis      -> outputs/results/analysis.json and analysis.md
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from fairreturn import config as cfg
from fairreturn.evaluation import metrics as M

HEADLINE_MODELS = ["gbdt_aware", "gbdt_unaware", "gbdt_blind", "gbdt_reweigh", "mlp", "vib_grl", "vib_grl_proxy"]


def _preds(name):
    f = cfg.PREDS_DIR / f"{name}.npz"
    return np.load(f) if f.exists() else None


def _logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def postage_effect(test: pd.DataFrame, n: int = 30000, seed: int = cfg.SEED) -> list[dict]:
    """Risk change when only the postage line is added (0 -> 1), for every saved model that uses it."""
    from fairreturn.data.features import FEATURE_SETS
    from fairreturn.models.registry import METHODS, LoadedModel
    sample = test.sample(min(n, len(test)), random_state=seed)
    rows = []
    for m in METHODS:
        if "has_postage" not in FEATURE_SETS[m.feature_set] or not (cfg.MODELS_DIR / m.name).exists():
            continue
        lm = LoadedModel(m.name)
        p0 = lm.predict(sample.assign(has_postage=0.0))
        p1 = lm.predict(sample.assign(has_postage=1.0))
        d = _logit(p1) - _logit(p0)
        rows.append({"method": m.name, "label": m.label, "role": m.role, "feature_set": m.feature_set,
                     "mean_delta_logit": float(d.mean()), "mean_abs_delta_pts": float(np.abs(p1 - p0).mean() * 100),
                     "share_lowered": float((p1 < p0).mean()), "lines": int(len(sample))})
    return rows


# Hand-written expert-system baseline: forward chaining over six if-then rules whose conditions were read off the
# TRAINING split (each condition at least 1.5x more or less likely to precede a return); the score is the sum of
# the points of the rules that fire. No country information is used.
def _rules(rbar: float):
    return [("the customer returned more than twice the average share of items before", 2,
             lambda d: d["cust_prior_return_rate"] > 2 * rbar),
            ("the product was returned more than twice the average share before", 2,
             lambda d: d["prod_prior_return_rate"] > 2 * rbar),
            ("the product is fragile", 1, lambda d: d["tag_fragile"] == 1),
            ("the basket has at most 5 different products", 1, lambda d: d["basket_lines"] <= 5),
            ("the line quantity is 24 or more", 1, lambda d: d["quantity"] >= 24),
            ("the basket has 50 or more different products", -1, lambda d: d["basket_lines"] >= 50)]


def rule_system(L: pd.DataFrame) -> dict:
    from scipy.stats import rankdata
    tr, va, te = (L[L["split"] == s] for s in ("train", "val", "test"))
    rules = _rules(float(tr["returned"].mean()))
    # points; ties (many lines share a points total) are broken by the product's past return rate (< 1 point)
    score = lambda d: (sum(w * np.asarray(f(d), dtype=float) for _, w, f in rules)  # noqa: E731
                       + d["prod_prior_return_rate"].to_numpy())
    y = te["returned"].to_numpy()
    pts_val, pts_te = score(va), score(te)
    thr = M.threshold_top_share(pts_val)
    gt = M.group_table(y, pts_te, te["group"], thr, te["customer_id"])
    gaps = M.fairness_gaps(gt)
    auc = lambda s_: M.accuracy_metrics(y, s_)["auc"]  # noqa: E731
    # Bayesian history score: rank average of the two Beta-smoothed past return rates (ranks taken on each split)
    hist = lambda d: rankdata(d["prod_prior_return_rate"]) / len(d) + rankdata(d["cust_prior_return_rate"]) / len(d)  # noqa: E731
    h_val, h_te = hist(va), hist(te)
    gh = M.fairness_gaps(M.group_table(y, h_te, te["group"], M.threshold_top_share(h_val), te["customer_id"]))
    return {"rules": [{"if": t, "points": w} for t, w, _ in rules],
            "history_score": {"auc": auc(h_te), "fpr_gap": gh["fpr_gap"], "tpr_gap": gh["tpr_gap"]},
            "auc_points": auc(pts_te), "flag_share": float((pts_te >= thr).mean()),
            "fpr_gap": gaps["fpr_gap"], "tpr_gap": gaps["tpr_gap"],
            "auc_product_history_only": auc(te["prod_prior_return_rate"].to_numpy()),
            "auc_customer_history_only": auc(te["cust_prior_return_rate"].to_numpy()),
            "auc_rank_average": auc(h_te)}


def main():
    L = pd.read_pickle(cfg.LINES_FILE)
    test = L[L["split"] == "test"].reset_index(drop=True)
    y = test["returned"].to_numpy()
    out: dict = {}

    # 1. product tags ---------------------------------------------------------------------------------
    a, b = _preds("gbdt_blind"), _preds("gbdt_notags")
    if a is not None and b is not None:
        rows = []
        for name, mask in (("all test lines", np.ones(len(test), bool)),
                           ("products with < 50 earlier sales", test["prod_prior_lines"].to_numpy() < 50),
                           ("products with < 10 earlier sales", test["prod_prior_lines"].to_numpy() < 10),
                           ("fragile products", test["tag_fragile"].to_numpy() == 1)):
            if mask.sum() > 50 and len(np.unique(y[mask])) > 1:
                rows.append({"subset": name, "lines": int(mask.sum()), "returns": int(y[mask].sum()),
                             "auc_with_tags": M.accuracy_metrics(y[mask], a["test"][mask])["auc"],
                             "auc_without_tags": M.accuracy_metrics(y[mask], b["test"][mask])["auc"]})
        out["tag_value"] = rows

    # 2. group tables ---------------------------------------------------------------------------------
    headline = {}
    for name in HEADLINE_MODELS:
        p = _preds(name)
        if p is None:
            continue
        thr = M.threshold_top_share(p["val"])
        gt = M.group_table(y, p["test"], test["group"], thr, test["customer_id"])
        headline[name] = gt.set_index("group")[["fpr", "tpr", "flag_rate", "mean_score", "observed_rate",
                                                 "calibration_gap"]].round(4).to_dict(orient="index")
    out["group_tables"] = headline

    # 3. postage proxy --------------------------------------------------------------------------------
    try:
        out["postage_effect"] = postage_effect(test)
    except Exception as e:  # noqa: BLE001 - models missing or not loadable: skip this part
        out["postage_effect"] = []
        print(f"[warn] postage analysis skipped: {e}")

    # 4. rule-based expert system ----------------------------------------------------------------------
    out["rule_system"] = rule_system(L)

    # 5. sentences ------------------------------------------------------------------------------------
    sentences = []
    h = headline
    if "gbdt_aware" in h and "vib_grl" in h:
        out["headline"] = {
            "baseline_fpr_germany": h["gbdt_aware"]["Germany"]["fpr"], "baseline_fpr_uk": h["gbdt_aware"]["United Kingdom"]["fpr"],
            "ours_fpr_germany": h["vib_grl"]["Germany"]["fpr"], "ours_fpr_uk": h["vib_grl"]["United Kingdom"]["fpr"],
        }
        sentences.append(
            f"The country-aware baseline flags {h['gbdt_aware']['Germany']['fpr']:.0%} of German order lines that were "
            f"never returned, against {h['gbdt_aware']['United Kingdom']['fpr']:.0%} for UK lines; with VIB + GRL the "
            f"figures are {h['vib_grl']['Germany']['fpr']:.0%} and {h['vib_grl']['United Kingdom']['fpr']:.0%}.")
    pe = {r["method"]: r for r in out["postage_effect"]}
    if "vib_grl_proxy" in pe:
        others = [r for r in out["postage_effect"] if r["method"] != "vib_grl_proxy" and r["feature_set"] == "unaware"]
        ref = ", ".join(f"{r['label']} {r['mean_delta_logit']:+.2f}" for r in others)
        sentences.append(
            f"Adding only the postage line changes the log-odds of a return by {pe['vib_grl_proxy']['mean_delta_logit']:+.2f} "
            f"on average for VIB + GRL when the proxy is left in ({pe['vib_grl_proxy']['share_lowered']:.0%} of lines get a "
            f"lower risk), against {ref} for the models without an adversary: the adversary makes the encoder use the "
            "proxy in reverse. Our model therefore never receives the postage line.")
    out["sentences"] = sentences

    cfg.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (cfg.RESULTS_DIR / "analysis.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    md = ["# Extra analyses", ""]
    for s in sentences:
        md += [s, ""]
    if out.get("tag_value"):
        md += ["## Value of the product tags (GBDT-Blind with vs without tags)", "",
               "| Subset | Lines | Returns | AUC with tags | AUC without tags |", "|---|---|---|---|---|"]
        md += [f"| {r['subset']} | {r['lines']:,} | {r['returns']:,} | {r['auc_with_tags']:.4f} | {r['auc_without_tags']:.4f} |"
               for r in out["tag_value"]]
        md.append("")
    if out.get("postage_effect"):
        md += ["## Effect of adding only the postage line", "",
               "| Model | Inputs | Mean Δ log-odds | Mean abs Δ risk (pts) | Lines made less risky |", "|---|---|---|---|---|"]
        md += [f"| {r['label']} | {r['feature_set']} | {r['mean_delta_logit']:+.3f} | {r['mean_abs_delta_pts']:.2f} | "
               f"{r['share_lowered']:.0%} |" for r in out["postage_effect"]]
        md.append("")
    rs = out.get("rule_system")
    if rs:
        md += ["## Rule-based expert system (no learning, no country)", "",
               "| Rule (IF ...) | Points |", "|---|---|"] + [f"| {r['if']} | {r['points']:+d} |" for r in rs["rules"]]
        md += ["", f"Test AUC of the points score (ties broken by the product's past return rate) {rs['auc_points']:.3f} (flags {rs['flag_share']:.1%} of lines at the "
               f"validation top-10% threshold; ΔFPR {rs['fpr_gap']:.3f}, ΔTPR {rs['tpr_gap']:.3f}). Single-feature "
               f"scores: product history {rs['auc_product_history_only']:.3f}, customer history "
               f"{rs['auc_customer_history_only']:.3f}, rank average of the two {rs['auc_rank_average']:.3f} "
               f"(ΔFPR {rs['history_score']['fpr_gap']:.3f}, ΔTPR {rs['history_score']['tpr_gap']:.3f}).", ""]
    if headline:
        md += ["## False-alarm rate by country group (test, seed 0)", "",
               "| Model | " + " | ".join(cfg.GROUP_SHORT[g] for g in cfg.GROUPS) + " |", "|---|" + "---|" * len(cfg.GROUPS)]
        for name, t in headline.items():
            md.append(f"| {name} | " + " | ".join(f"{t[g]['fpr']:.1%}" if g in t else "—" for g in cfg.GROUPS) + " |")
    (cfg.RESULTS_DIR / "analysis.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
