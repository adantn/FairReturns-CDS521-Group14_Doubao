"""Evaluate the notes written by the decision assistant (part of 'Evaluation of generated contents').

For N real test order lines - half that our model flags as high-risk, half that it does not - the assistant
is asked for a next step exactly as in the web app. Measured on the generated answers:

    llm_answer_rate        calls that returned usable JSON (otherwise the rule-based note is shown)
    guardrail_pass_rate    LLM answers that passed the guardrail unchanged
    replaced_country       ... replaced because the note mentioned a country, nationality or postage
    replaced_action        ... replaced because the action was not one of the allowed actions
    replaced_penalising    ... replaced because the note suggested a surcharge, deposit or refusal
    action_agreement       LLM action = action of the rule-based policy (plus Cohen's kappa)
    threshold_consistency  'no_action' exactly when the risk is below the high-risk threshold
    cites_fact_rate        notes that refer to at least one order fact (customer, product, quantity ...)
    unsupported_number_rate  numbers in the note that do not appear in the prompt (a simple hallucination check)

    python -m fairreturn.genai.assistant_eval --backend api --n 100     # real run: N LLM calls
    python -m fairreturn.genai.assistant_eval --backend mock            # pipeline test, numbers NOT reportable

Outputs: outputs/genai/assistant_eval/results.csv and summary.json
"""
from __future__ import annotations

import argparse
import json
import re

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

from fairreturn import config as cfg
from fairreturn.evaluation import explain
from fairreturn.genai import assistant, prompts
from fairreturn.genai.llm_client import LLMClient

OUT_DIR = cfg.GENAI_DIR / "assistant_eval"
NUMBER = re.compile(r"\d+(?:[.,]\d+)?%?")


def numbers_not_in_prompt(message: str, prompt: str) -> tuple[int, int]:
    """(numbers in the note with at least two characters, how many of them do not occur in the prompt)."""
    nums = [n for n in NUMBER.findall(message) if len(n) >= 2]
    missing = [n for n in nums if n.rstrip("%") not in prompt]
    return len(nums), len(missing)


def sample_lines(n: int, seed: int = cfg.SEED):
    from fairreturn.models.registry import LoadedModel
    lm = LoadedModel(cfg.DEMO_OURS)
    thr = float(lm.info["metrics"]["threshold"])
    test = pd.read_pickle(cfg.LINES_FILE)
    test = test[test["split"] == "test"].groupby("invoice", sort=False).head(1)   # one line per invoice
    risk = lm.predict(test)
    rng = np.random.default_rng(seed)
    hi, lo = np.flatnonzero(risk >= thr), np.flatnonzero(risk < thr)
    pick = np.concatenate([rng.choice(hi, size=min(len(hi), n // 2), replace=False),
                           rng.choice(lo, size=min(len(lo), n - n // 2), replace=False)])
    rows = test.iloc[pick].reset_index(drop=True)
    X = lm.matrix(rows)
    contrib = explain.contributions(lm.predict_matrix, X, lm.fb, lm.reference)
    return rows, lm.predict_matrix(X), [explain.top_factors(c, k=5) for c in contrib], thr


def run(backend: str, n: int, seed: int = cfg.SEED) -> dict:
    client = LLMClient(backend=backend, cache_file=OUT_DIR / "llm_cache.jsonl")
    rows, risk, factors, thr = sample_lines(n, seed)
    recs = []
    for i, (_, row) in enumerate(rows.iterrows()):
        prompt = assistant.build_user_prompt(row, float(risk[i]), thr, factors[i])
        ans = assistant.assist(row, float(risk[i]), thr, factors[i], client, allow_mock=True)
        rule = assistant.rule_based(row, float(risk[i]), thr, factors[i])
        llm_text = ans.get("llm_raw") or (ans["message"] if ans["source"].startswith("llm:") else "")
        n_num, n_missing = numbers_not_in_prompt(llm_text, prompt) if llm_text else (0, 0)
        recs.append({
            "line_id": row["line_id"], "risk": float(risk[i]), "high_risk": bool(risk[i] >= thr),
            "source": ans["source"], "guardrail": ans["guardrail"], "action": ans["action"],
            "rule_action": rule["action"], "llm_answered": bool(llm_text),
            "replaced": ans["guardrail"].startswith("LLM answer replaced"),
            "replaced_country": "mentioned country" in ans["guardrail"],
            "replaced_action": "not allowed" in ans["guardrail"],
            "replaced_penalising": "penalising" in ans["guardrail"],
            "cites_fact": bool(prompts.mentioned_facts(llm_text)) if llm_text else False,
            "numbers": n_num, "numbers_not_in_prompt": n_missing, "message": ans["message"], "llm_text": llm_text,
        })
    df = pd.DataFrame(recs)
    answered = df[df["llm_answered"]]
    passed = df[df["source"].str.startswith("llm:")]
    summary = {
        "backend": client.backend, "model": client.model, "is_mock": client.is_mock, "n": int(len(df)),
        "threshold": thr,
        "llm_answer_rate": float(df["llm_answered"].mean()),
        "guardrail_pass_rate": float(len(passed) / max(1, len(answered))),
        "replaced_country": int(answered["replaced_country"].sum()),
        "replaced_action": int(answered["replaced_action"].sum()),
        "replaced_penalising": int(answered["replaced_penalising"].sum()),
        "action_agreement": float((passed["action"] == passed["rule_action"]).mean()) if len(passed) else None,
        "action_kappa": (float(cohen_kappa_score(passed["action"], passed["rule_action"]))
                         if len(passed) and len(set(passed["action"]) | set(passed["rule_action"])) > 1 else None),
        "threshold_consistency": (float(((passed["action"] == "no_action") == (~passed["high_risk"])).mean())
                                  if len(passed) else None),
        "cites_fact_rate": float(answered["cites_fact"].mean()) if len(answered) else None,
        "unsupported_number_rate": (float(answered["numbers_not_in_prompt"].sum() / answered["numbers"].sum())
                                    if len(answered) and answered["numbers"].sum() else 0.0),
        "action_distribution": passed["action"].value_counts().to_dict(),
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_DIR / "results.csv", index=False)
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backend", choices=["auto", "api", "mock"], default="auto")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=cfg.SEED)
    a = ap.parse_args()
    s = run(a.backend, a.n, a.seed)
    print(json.dumps(s, indent=2))
    if s["is_mock"]:
        print("\nMOCK backend: these numbers only show that the evaluation runs. Do not report them.")
    print(f"wrote {OUT_DIR / 'results.csv'} and summary.json")


if __name__ == "__main__":
    main()
