"""GenAI module tests (offline: mock backend, no network)."""
from __future__ import annotations

import json

import pandas as pd

from fairreturn.genai import assistant, prompts
from fairreturn.genai.llm_client import LLMClient, extract_json
from fairreturn.genai.product_tagger import CATEGORIES, MATERIALS, rule_tag, tag_products_with_llm

ROW = pd.Series({
    "description": "SET OF 3 GLASS STORAGE JARS", "quantity": 12, "price": 2.5, "product_median_price": 2.95,
    "basket_lines": 30, "basket_value": 410.0, "basket_qty": 300, "has_postage": 1, "cust_prior_invoices": 5,
    "cust_prior_lines": 120, "cust_prior_returns": 6, "cust_prior_spend": 2100.0, "cust_tenure_days": 240,
    "cust_recency_days": 21, "prod_prior_lines": 80, "prod_prior_returns": 4,
    "invoice_date": pd.Timestamp("2011-11-02 10:15"), "tag_category": "kitchen_dining", "tag_material": "glass",
    "tag_fragile": 1, "tag_seasonal": 0, "group": "Germany", "country": "Germany",
})


def test_rule_tagger_basics():
    t = rule_tag("WHITE HANGING HEART T-LIGHT HOLDER")
    assert t["category"] == "candles_lighting" and not t["fragile"]
    g = rule_tag("SET/5 RED RETROSPOT LID GLASS BOWLS")
    assert g == {"category": "kitchen_dining", "material": "glass", "fragile": True, "seasonal": False}
    assert rule_tag("PAPER CHAIN KIT 50'S CHRISTMAS")["seasonal"]
    for d in ("JUMBO BAG RED RETROSPOT", "ZINC METAL HEART DECORATION", "???"):
        t = rule_tag(d)
        assert t["category"] in CATEGORIES and t["material"] in MATERIALS


def test_llm_tagger_with_mock_validates_and_falls_back():
    products = pd.DataFrame({"stock_code": [f"S{i}" for i in range(60)],
                             "description": [f"GLASS VASE NUMBER {i}" if i % 2 else f"WOODEN SIGN {i}" for i in range(60)]})
    out = tag_products_with_llm(products, LLMClient(backend="mock"), batch_size=25)
    assert len(out) == 60 and out["tag_category"].isin(CATEGORIES).all()
    assert out["tag_source"].str.endswith("_mock").all()
    invalid = out[out["llm_valid"] == 0]
    assert (invalid["tag_source"] == "rules_fallback_mock").all()


def test_extract_json_variants():
    assert extract_json('```json\n{"risk": 12, "reason": "x"}\n```', prefer="object")["risk"] == 12
    assert extract_json('Sure! [{"id": 1}] done')[0]["id"] == 1
    assert extract_json('{"risk": 5, "reason": "see [note]"}', prefer="object")["risk"] == 5


def test_prompt_never_leaks_country_in_tool_condition():
    from fairreturn.genai.llm_audit import build_prompt
    plain = build_prompt(ROW, "plain")
    masked = build_prompt(ROW, "masked")
    tool = build_prompt(ROW, "tool", 0.031, ["customer returned 6 of 120 earlier items (raises risk)"])
    assert "Customer's country: Germany" in plain and "Postage charged" in plain
    assert "Customer's country" not in masked and "Postage charged" in masked
    assert "Customer's country" not in tool and "Postage" not in tool and "3.1%" in tool


def test_country_word_detector():
    assert prompts.mentions_country("High risk because the customer is from Germany.")
    assert prompts.mentions_country("International postage adds risk")
    assert not prompts.mentions_country("The customer returned 6 of 120 items before.")


def test_assistant_guardrail_replaces_biased_answer():
    class Fake(LLMClient):
        def __init__(self, reply):
            super().__init__(backend="mock")
            self.backend, self.model, self._reply = "api", "fake", reply

        def chat(self, system, user, mock_fn=None, meta=None):
            assert "Germany" not in user and "Postage" not in user     # the assistant never sees country
            return self._reply

    factors = [("Customer history", 0.01), ("Product attributes", 0.004)]
    biased = Fake(json.dumps({"action": "manual_review", "message": "German customers return a lot."}))
    r = assistant.assist(ROW, 0.05, 0.03, factors, biased)
    assert r["source"] == "rules" and "replaced" in r["guardrail"]
    clean = Fake(json.dumps({"action": "packaging_check", "message": "Glass item; check the packing."}))
    r2 = assistant.assist(ROW, 0.05, 0.03, factors, clean)
    assert r2["source"] == "llm:fake" and r2["guardrail"] == "passed"
    bad_action = Fake(json.dumps({"action": "add_surcharge", "message": "Charge a deposit."}))
    assert assistant.assist(ROW, 0.05, 0.03, factors, bad_action)["source"] == "rules"


def test_assistant_eval_helpers_and_mock_guardrail():
    from fairreturn.genai.assistant_eval import numbers_not_in_prompt
    assert numbers_not_in_prompt("Risk 6.1% above 5.8%; about 37% come back.", "risk 6.1% threshold 5.8%") == (3, 1)
    mock = LLMClient(backend="mock")
    factors = [("Customer history", 0.01), ("Product attributes", 0.004)]
    seen = {"passed": 0, "replaced": 0}
    for q in range(1, 40):                       # different prompts -> the mock sometimes breaks the rules
        row = {**ROW, "quantity": q}
        r = assistant.assist(row, 0.08, 0.05, factors, mock, allow_mock=True)
        assert r["action"] in assistant.ACTIONS and not prompts.mentions_country(r["message"])
        seen["passed" if r["guardrail"] == "passed" else "replaced"] += 1
    assert seen["passed"] and seen["replaced"]  # both paths were exercised
    # without allow_mock the web app keeps the rule-based note
    assert assistant.assist(ROW, 0.08, 0.05, factors, mock)["source"] == "rules"
