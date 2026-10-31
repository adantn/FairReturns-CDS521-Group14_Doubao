"""Every number the report leaves as a yellow placeholder (⟦T3⟧, ⟦A2⟧, ...) - computed from the result files of
the real-LLM runs, and optionally written into the Word file.

The placeholders exist because these results need a real LLM (the offline mock backend is only a pipeline test):

    python -m fairreturn.genai.product_tagger --backend api --top 1000              # T: product tags
    python -m fairreturn.experiments.run_all --product-tags llm --methods gbdt_blind gbdt_notags vib_grl \\
           --seeds 3 --bootstrap 0 --quiet                                         # P: value of the LLM tags
    python -m fairreturn.genai.llm_audit --backend api                              # A: LLM audit
    python -m fairreturn.genai.assistant_eval --backend api --n 100                 # S: decision assistant
    (two teammates fill data/processed/human_tag_sample.csv, then rerun the tagger)  # H: human check

    python -m fairreturn.experiments.report_numbers                                 # print all values
    python -m fairreturn.experiments.report_numbers --fill report.docx              # -> report_filled.docx

--fill replaces each placeholder whose value is available and removes its highlight; placeholders without a
value stay yellow, so nothing unfinished can slip through. Numbers from the mock backend are refused.
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from fairreturn import config as cfg

CONDITIONS = ["plain", "instructed", "masked", "tool"]
AUDIT_COLS = [("auc", "AUC", "f3"), ("cf_flip_rate", "decisions that flip", "pct"),
              ("mean_abs_delta_risk", "mean |change in risk| (points)", "f1"),
              ("foreign_surcharge", "risk added when a UK order moves abroad (points)", "signed1"),
              ("mentions_country_rate", "reasons mentioning country/postage", "pct"),
              ("cites_fact_rate", "reasons citing an order fact", "pct")]
FIELDS = ["category", "material", "fragile", "seasonal"]


def _fmt(v, kind: str) -> str | None:
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return None
    if kind == "pct":
        return f"{100 * float(v):.1f}%"
    if kind == "f3":
        return f"{float(v):.3f}"
    if kind == "f2":
        return f"{float(v):.2f}"
    if kind == "f1":
        return f"{float(v):.1f}"
    if kind == "signed1":
        return f"{float(v):+.1f}"
    if kind == "int":
        return f"{int(v):,}"
    return str(v)


def _read_json(p: Path):
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def collect(allow_mock: bool = False) -> tuple[list[tuple[str, str | None, str]], list[str]]:
    """[(code, value or None, description)], [warnings]"""
    out, warn = [], []

    def add(code, value, kind, desc):
        out.append((code, _fmt(value, kind) if value is not None else None, desc))

    # ---------------------------------------------------------------- T: product tags
    ev = _read_json(cfg.GENAI_DIR / "tag_eval.json")
    if ev and ev.get("is_mock") and not allow_mock:
        warn.append("tag_eval.json comes from the MOCK backend: T and H values left empty")
        ev = None
    add("T1", (ev or {}).get("llm", {}).get("model") if ev else None, "str", "LLM used for product tags")
    add("T2", ev.get("n_products_tagged") if ev else None, "int", "products sent to the LLM")
    add("T3", ev.get("validity") if ev else None, "pct", "answers using only allowed values")
    add("T4", ev.get("coverage_lines_valid_llm") if ev else None, "pct", "purchase lines covered by valid LLM tags")
    for i, f in enumerate(FIELDS):
        m = (ev or {}).get("llm_vs_rules", {}).get(f, {})
        add(f"T{5 + i}", m.get("accuracy"), "pct", f"LLM vs keyword rules, {f}: agreement")
        add(f"T{9 + i}", m.get("kappa"), "f2", f"LLM vs keyword rules, {f}: Cohen's kappa")

    # ---------------------------------------------------------------- H: human check (60 products)
    hum = (ev or {}).get("human", {}) if ev else {}
    ok = hum.get("status") == "ok"
    add("H0", hum.get("n_labelled") if ok else None, "int", "products labelled by two teammates")
    for i, f in enumerate(FIELDS):
        r = hum.get(f, {}) if ok else {}
        add(f"H{1 + 3 * i}", (r.get("llm_vs_A") or {}).get("accuracy"), "pct", f"LLM vs annotator A, {f}")
        add(f"H{2 + 3 * i}", (r.get("rules_vs_A") or {}).get("accuracy"), "pct", f"rules vs annotator A, {f}")
        add(f"H{3 + 3 * i}", (r.get("A_vs_B") or {}).get("kappa"), "f2", f"annotator A vs B, {f}: kappa")

    # ---------------------------------------------------------------- P: value of the LLM tags in the models
    P = cfg.RESULTS_DIR / "llm_tags"
    summ = pd.read_csv(P / "summary.csv") if (P / "summary.csv").exists() else None
    preds = P / "preds" / "gbdt_blind.npz"
    if summ is not None or preds.exists():
        if ev is None and not allow_mock:
            warn.append("LLM-tag experiment found but the tags are mock or missing: P values left empty")
            summ, preds = None, Path("/nonexistent")
    vals = {}
    if preds.exists() and cfg.LINES_FILE.exists():
        from fairreturn.evaluation import metrics as M
        L = pd.read_pickle(cfg.LINES_FILE)
        test = L[L["split"] == "test"].reset_index(drop=True)
        y, p = test["returned"].to_numpy(), np.load(preds)["test"]
        for key, mask in (("all", np.ones(len(test), bool)),
                          ("lt50", test["prod_prior_lines"].to_numpy() < 50),
                          ("lt10", test["prod_prior_lines"].to_numpy() < 10)):
            vals[key] = M.accuracy_metrics(y[mask], p[mask])["auc"]
    add("P1", vals.get("all"), "f3", "GBDT-Blind with LLM tags: AUC, all test lines")
    add("P2", vals.get("lt50"), "f3", "GBDT-Blind with LLM tags: AUC, products with < 50 earlier sales")
    add("P3", vals.get("lt10"), "f3", "GBDT-Blind with LLM tags: AUC, products with < 10 earlier sales")
    row = summ.set_index("method").loc["vib_grl"] if summ is not None and "vib_grl" in set(summ["method"]) else None
    add("P4", None if row is None else row["auc_mean"], "f3", "VIB + GRL with LLM tags: AUC (mean over seeds)")
    add("P5", None if row is None else row["fpr_gap_mean"], "f3", "VIB + GRL with LLM tags: ΔFPR (mean over seeds)")

    # ---------------------------------------------------------------- A: LLM audit
    s = cfg.AUDIT_DIR / "summary.csv"
    aud = pd.read_csv(s) if s.exists() else None
    if aud is not None and aud["is_mock"].any() and not allow_mock:
        warn.append("audit summary comes from the MOCK backend: A values left empty")
        aud = None
    add("A0", None if aud is None else str(aud["model"].iloc[0]), "str", "LLM audited")
    add("A00", None if aud is None else int(aud["pairs"].iloc[0]), "int", "audited pairs (order + twin)")
    k = 1
    for c in CONDITIONS:
        r = None if aud is None or c not in set(aud["condition"]) else aud.set_index("condition").loc[c]
        for col, desc, kind in AUDIT_COLS:
            add(f"A{k}", None if r is None else r[col], kind, f"audit, {c} prompt: {desc}")
            k += 1
    add("A25", None if aud is None else float(aud["parse_ok_rate"].min()), "pct", "audit: lowest share of parseable answers")

    # ---------------------------------------------------------------- S: decision assistant
    sa = _read_json(cfg.GENAI_DIR / "assistant_eval" / "summary.json")
    if sa and sa.get("is_mock") and not allow_mock:
        warn.append("assistant_eval comes from the MOCK backend: S values left empty")
        sa = None
    g = (lambda key: None if sa is None else sa.get(key))
    add("S1", g("model"), "str", "LLM used by the decision assistant")
    add("S2", g("n"), "int", "order lines evaluated")
    add("S3", g("llm_answer_rate"), "pct", "calls with a usable LLM answer")
    add("S4", g("guardrail_pass_rate"), "pct", "LLM answers that passed the guardrail")
    add("S5", g("replaced_country"), "int", "answers replaced: mentioned country / postage")
    add("S6", g("replaced_action"), "int", "answers replaced: action not allowed")
    add("S7", g("replaced_penalising"), "int", "answers replaced: penalising wording")
    add("S8", g("action_agreement"), "pct", "LLM action = rule-based policy")
    add("S9", g("action_kappa"), "f2", "LLM action vs rule-based policy: kappa")
    add("S10", g("threshold_consistency"), "pct", "'no action' exactly below the threshold")
    add("S11", g("cites_fact_rate"), "pct", "notes citing an order fact")
    add("S12", g("unsupported_number_rate"), "pct", "numbers in notes that are not in the prompt")
    return out, warn


# ----------------------------------------------------------------------------- docx filling
RUN = re.compile(r"<w:r>(?:(?!</w:r>).)*?</w:r>|<w:r (?:(?!</w:r>).)*?</w:r>", re.S)
TOKEN = re.compile(r"⟦([A-Z]\d+)⟧")


def fill_docx(src: Path, dst: Path, values: dict[str, str]) -> tuple[int, set[str]]:
    """Replace ⟦CODE⟧ tokens in the document body; returns (replacements, codes still open)."""
    with zipfile.ZipFile(src) as z:
        xml = z.read("word/document.xml").decode("utf-8")
    count, still_open = 0, set()

    def fix_run(m):
        nonlocal count
        run = m.group(0)
        codes = TOKEN.findall(run)
        if not codes:
            return run
        if all(c in values for c in codes):
            for c in codes:
                run = run.replace(f"⟦{c}⟧", _xml_escape(values[c]))
                count += 1
            run = re.sub(r'<w:highlight w:val="yellow"\s*/>', "", run)
        else:
            still_open.update(c for c in codes if c not in values)
        return run

    xml = RUN.sub(fix_run, xml)
    tmp = Path(tempfile.mkdtemp()) / "out.docx"
    with zipfile.ZipFile(src) as zin, zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as zout:
        for item in zin.infolist():
            data = xml.encode("utf-8") if item.filename == "word/document.xml" else zin.read(item.filename)
            zout.writestr(item, data)
    shutil.move(str(tmp), dst)
    return count, still_open


def _xml_escape(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fill", type=Path, default=None, help="report .docx with ⟦CODE⟧ placeholders")
    ap.add_argument("--out", type=Path, default=None, help="output file (default: <name>_filled.docx)")
    ap.add_argument("--allow-mock", action="store_true", help="pipeline test only: accept mock numbers")
    a = ap.parse_args()
    rows, warn = collect(allow_mock=a.allow_mock)
    width = max(len(c) for c, _, _ in rows)
    for code, value, desc in rows:
        print(f"  ⟦{code}⟧{' ' * (width - len(code))}  {value if value is not None else '—':>14}   {desc}")
    for w in warn:
        print("  [!] " + w)
    missing = [c for c, v, _ in rows if v is None]
    print(f"\n{len(rows) - len(missing)} of {len(rows)} values available.")
    if a.fill:
        values = {c: v for c, v, _ in rows if v is not None}
        out = a.out or a.fill.with_name(a.fill.stem + "_filled.docx")
        n, still = fill_docx(a.fill, out, values)
        print(f"filled {n} placeholders -> {out}" + (f"; still open: {', '.join(sorted(still))}" if still else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
