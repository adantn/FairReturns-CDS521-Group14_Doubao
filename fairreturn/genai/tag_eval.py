"""Evaluate LLM-generated product tags (the 'Evaluation of generated contents' part of the project).

Reported:
  * validity     - share of LLM answers that used only allowed values (before falling back to rules)
  * coverage     - share of products / purchase lines that received an LLM tag
  * agreement    - LLM vs keyword rules, per attribute (accuracy and Cohen's kappa)
  * human check  - if data/processed/human_tag_sample.csv has been filled in by two teammates:
                   accuracy and kappa of LLM and rules against annotator A, plus A-vs-B agreement
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

from fairreturn import config as cfg
from fairreturn.genai.product_tagger import CATEGORIES, MATERIALS, rule_tag

FIELDS = ["category", "material", "fragile", "seasonal"]


def _agree(a, b) -> dict:
    a, b = pd.Series(a).astype(str).to_numpy(), pd.Series(b).astype(str).to_numpy()
    if len(a) == 0:
        return {"n": 0, "accuracy": None, "kappa": None}
    acc = float(np.mean(a == b))
    try:
        kappa = float(cohen_kappa_score(a, b)) if len(set(a) | set(b)) > 1 else 1.0
    except ValueError:
        kappa = None
    return {"n": int(len(a)), "accuracy": round(acc, 4), "kappa": None if kappa is None else round(kappa, 4)}


def evaluate(products: pd.DataFrame, llm_tags: pd.DataFrame, client_info: dict | None = None) -> dict:
    llm = llm_tags.set_index("stock_code")
    rules = {sc: rule_tag(d) for sc, d in zip(llm.index, llm["description"])}
    report: dict = {"llm": client_info or {}, "n_products_tagged": int(len(llm)),
                    "is_mock": bool(str((client_info or {}).get("backend", "")).startswith("mock"))}
    report["validity"] = round(float(llm["llm_valid"].mean()), 4) if len(llm) else None
    problems = llm.loc[llm["llm_valid"] == 0, "llm_problem"].value_counts().head(10)
    report["invalid_examples"] = {str(k): int(v) for k, v in problems.items()}
    total_lines = float(products["n_lines"].sum())
    covered = products[products["stock_code"].isin(llm.index[llm["llm_valid"] == 1])]["n_lines"].sum()
    report["coverage_products"] = round(len(llm) / max(1, len(products)), 4)
    report["coverage_lines_valid_llm"] = round(float(covered) / max(1.0, total_lines), 4)

    valid = llm[llm["llm_valid"] == 1]
    report["llm_vs_rules"] = {}
    for f in FIELDS:
        col = f"tag_{f}"
        ref = [rules[sc][f] if f in ("category", "material") else int(rules[sc][f]) for sc in valid.index]
        report["llm_vs_rules"][f] = _agree(valid[col], ref)
    report["category_distribution"] = {
        "llm": valid["tag_category"].value_counts().to_dict(),
        "rules": pd.Series([rules[sc]["category"] for sc in valid.index]).value_counts().to_dict(),
    }
    disagreements = []
    for sc, row in valid.iterrows():
        r = rules[sc]
        if row["tag_category"] != r["category"] or row["tag_material"] != r["material"]:
            disagreements.append({"stock_code": sc, "description": row["description"],
                                  "llm": f"{row['tag_category']} / {row['tag_material']}",
                                  "rules": f"{r['category']} / {r['material']}"})
    report["disagreement_examples"] = disagreements[:15]
    report["human"] = evaluate_human(llm)
    cfg.GENAI_DIR.mkdir(parents=True, exist_ok=True)
    (cfg.GENAI_DIR / "tag_eval.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def ensure_human_template(products: pd.DataFrame, n: int = 60, seed: int = cfg.SEED) -> None:
    """Write a blank annotation sheet once (sampled across popularity levels)."""
    path = cfg.HUMAN_TAG_TEMPLATE
    if path.exists():
        return
    top = products.sort_values("n_lines", ascending=False).head(1000)
    sample = top.sample(min(n, len(top)), random_state=seed)[["stock_code", "description"]]
    for who in ("a", "b"):
        for f in FIELDS:
            sample[f"{who}_{f}"] = ""
    path.parent.mkdir(parents=True, exist_ok=True)
    sample.to_csv(path, index=False, encoding="utf-8-sig")
    (path.parent / "human_tag_instructions.txt").write_text(
        "Two teammates label each product independently (columns a_* and b_*), without looking at the model tags.\n"
        f"category: one of {CATEGORIES}\nmaterial: one of {MATERIALS}\n"
        "fragile: 1 if likely to break in transit, else 0\nseasonal: 1 if meant for a holiday/season, else 0\n"
        "Then re-run: python -m fairreturn.genai.product_tagger (cached LLM answers are reused).\n",
        encoding="utf-8")


def evaluate_human(llm: pd.DataFrame) -> dict:
    path = cfg.HUMAN_TAG_TEMPLATE
    if not path.exists():
        return {"status": "template not created yet"}
    h = pd.read_csv(path, dtype=str, keep_default_na=False)
    filled = h[(h["a_category"].str.strip() != "")]
    if filled.empty:
        return {"status": f"waiting for annotations in {path.name}"}
    out = {"status": "ok", "n_labelled": int(len(filled))}
    for f in FIELDS:
        a = filled[f"a_{f}"].str.strip().str.lower()
        rules = [str(int(rule_tag(d)[f])) if f in ("fragile", "seasonal") else rule_tag(d)[f]
                 for d in filled["description"]]
        res = {"rules_vs_A": _agree(rules, a)}
        inter = filled["stock_code"].isin(llm.index)
        if inter.any():
            sub = filled[inter]
            col = f"tag_{f}"
            llm_vals = llm.loc[sub["stock_code"], col].astype(str).str.lower().to_numpy()
            res["llm_vs_A"] = _agree(llm_vals, sub[f"a_{f}"].str.strip().str.lower())
        b = filled[f"b_{f}"].str.strip().str.lower()
        if (b != "").all():
            res["A_vs_B"] = _agree(a, b)
        out[f] = res
    return out


def format_report(r: dict) -> str:
    lines = [f"Tag evaluation ({r['llm'].get('backend')}:{r['llm'].get('model')})"
             + ("  [MOCK - pipeline test only]" if r.get("is_mock") else ""),
             f"  valid LLM answers: {r['validity']:.1%}   products tagged: {r['n_products_tagged']}   "
             f"purchase lines covered: {r['coverage_lines_valid_llm']:.1%}"]
    for f, m in r["llm_vs_rules"].items():
        lines.append(f"  LLM vs rules - {f:9s} accuracy {m['accuracy']}  kappa {m['kappa']}")
    lines.append(f"  human check: {r['human'].get('status')}")
    return "\n".join(lines)
