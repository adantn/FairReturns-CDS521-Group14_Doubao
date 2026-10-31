"""Decision assistant: an LLM turns our debiased risk score into a fair next step for staff.

The LLM only ever sees order facts, our model's score and its main factors - never the country or the
postage line. Its answer then passes a guardrail: if it names a country, nationality or postage, or picks
an action outside the allowed list, it is replaced by a rule-based answer and the event is flagged.
Without an API key (or on any error) the same rule-based answer is used, so the demo always works.
"""
from __future__ import annotations

import hashlib
import json
import re

from fairreturn.genai import prompts
from fairreturn.genai.llm_client import LLMClient, extract_json

ACTIONS = {
    "no_action": "No action needed",
    "confirm_details": "Send the customer the product details and dimensions before dispatch",
    "packaging_check": "Check protective packaging before dispatch",
    "confirm_quantity": "Confirm the order quantity with the customer",
    "manual_review": "Ask a team member to review the order before dispatch",
}
SYSTEM = ("You help the customer-service team of a UK online wholesaler decide a fair next step for an order "
          "line with a predicted return risk. Rules: never use, mention or guess the customer's country, "
          "nationality, location or postage; never suggest surcharges, deposits, blocking or refusing the order. "
          "Answer with JSON only.")


def rule_based(row, risk: float, threshold: float, factors: list[tuple[str, float]]) -> dict:
    top = factors[0][0] if factors else ""
    if risk < threshold:
        action = "no_action"
    elif float(row["tag_fragile"]) and top in ("Product attributes", "Product history"):
        action = "packaging_check"
    elif top == "Order & basket":
        action = "confirm_quantity"
    elif top == "Product history":
        action = "confirm_details"
    else:
        action = "manual_review"
    reasons = prompts.factor_phrases(row, factors, k=2)
    msg = (f"Predicted return risk {prompts.pct(risk)} "
           + ("is below" if risk < threshold else "is above") + " the high-risk threshold. "
           + (f"Main factors: {'; '.join(reasons)}." if reasons else ""))
    return {"action": action, "action_label": ACTIONS[action], "message": msg.strip()}


def build_user_prompt(row, risk: float, threshold: float, factors: list[tuple[str, float]]) -> str:
    facts = prompts.order_facts(row)
    return ("Order line\n- " + "\n- ".join(facts)
            + f"\n- Debiased model: {prompts.pct(risk)} probability of return (high-risk threshold "
              f"{prompts.pct(threshold)}); main factors: {'; '.join(prompts.factor_phrases(row, factors)) or 'none'}"
            + f"\n\nChoose one action from {list(ACTIONS)} and write a two-sentence note for staff that "
              'explains it using only the facts above. Reply only with JSON: {"action": "...", "message": "..."}')


def mock_assistant_reply(system: str, user: str) -> str:
    """Offline stand-in used only by assistant_eval --backend mock: a deterministic note that sometimes breaks
    the rules on purpose (mentions a country, invents an action, invents a number) so every guardrail path runs."""
    m = re.search(r"Debiased model: ([\d.]+)% probability of return \(high-risk threshold ([\d.]+)%", user)
    risk, thr = (float(m.group(1)), float(m.group(2))) if m else (0.0, 1.0)
    h = int(hashlib.md5(user.encode()).hexdigest()[:6], 16)
    action = "no_action" if risk < thr else ("packaging_check" if "fragile: yes" in user else "manual_review")
    message = f"MOCK: predicted risk {risk:.1f}% against a threshold of {thr:.1f}%. The main factors are listed above."
    if h % 7 == 0:
        message += " The customer orders from abroad."            # must be caught by the guardrail
    if h % 11 == 0:
        action = "call_customer"                                   # not an allowed action
    if h % 13 == 0:
        message += " About 37% of similar orders come back."       # a number that is not in the prompt
    return json.dumps({"action": action, "message": message})


def assist(row, risk: float, threshold: float, factors: list[tuple[str, float]], client: LLMClient | None,
           allow_mock: bool = False) -> dict:
    """allow_mock=True lets the offline mock answer (pipeline tests only); the web app keeps the rule-based note."""
    fallback = rule_based(row, risk, threshold, factors)
    if client is None or (client.is_mock and not allow_mock):
        return {**fallback, "source": "rules", "guardrail": "not needed (rule-based answer)"}
    user = build_user_prompt(row, risk, threshold, factors)
    try:
        reply = client.chat(SYSTEM, user, mock_fn=mock_assistant_reply if client.is_mock else None,
                            meta={"task": "assistant"})
        obj = extract_json(reply, prefer="object")
        action, message = str(obj.get("action", "")).strip(), str(obj.get("message", "")).strip()
    except Exception as e:  # noqa: BLE001 - network, parsing: fall back, never crash the demo
        return {**fallback, "source": "rules", "guardrail": f"LLM unavailable ({type(e).__name__})"}
    problems = []
    if action not in ACTIONS:
        problems.append(f"action {action!r} not allowed")
    if prompts.mentions_country(message):
        problems.append("mentioned country / postage")
    if any(w in message.lower() for w in ("surcharge", "deposit", "refuse", "block", "decline the order")):
        problems.append("suggested a penalising action")
    if problems:
        return {**fallback, "source": "rules", "guardrail": "LLM answer replaced: " + "; ".join(problems),
                "llm_raw": message}
    return {"action": action, "action_label": ACTIONS[action], "message": message, "source": f"llm:{client.model}",
            "guardrail": "passed"}
