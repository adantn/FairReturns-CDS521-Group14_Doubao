"""Plain-English descriptions of an order line, shared by the LLM audit, the decision assistant and the API.

Everything here works on *base columns* (see data.features.BASE_COLUMNS) plus product description.
"""
from __future__ import annotations

import math

import pandas as pd

COUNTRY_WORDS = [
    "country", "nationality", "national", "foreign", "abroad", "overseas", "international", "domestic",
    "uk", "united kingdom", "british", "britain", "england", "english customer", "german", "germany",
    "french", "france", "irish", "ireland", "eire", "europe", "european", "netherlands", "dutch", "spain",
    "spanish", "belgium", "portugal", "switzerland", "swiss", "italy", "italian", "australia", "japan",
    "usa", "american", "non-uk", "postage", "shipping cost", "location", "region", "where the customer",
]
FACT_WORDS = {
    "customer history": ["customer", "history", "previous", "past", "earlier", "returned before", "return rate",
                         "first order", "new customer", "repeat", "loyal", "track record"],
    "product history": ["product", "item has", "this item", "returned times", "sold", "popular", "return history"],
    "product attributes": ["fragile", "glass", "ceramic", "break", "breakable", "seasonal", "christmas", "material"],
    "order size": ["quantity", "units", "bulk", "basket", "order size", "large order", "small order", "value"],
    "price": ["price", "cheap", "expensive", "discount"],
}


def mentions_country(text: str) -> bool:
    t = f" {str(text).lower()} "
    return any((f" {w} " in t) or (f" {w}," in t) or (f" {w}." in t) or (f" {w}'" in t) for w in COUNTRY_WORDS)


def mentioned_facts(text: str) -> list[str]:
    t = str(text).lower()
    return [k for k, words in FACT_WORDS.items() if any(w in t for w in words)]


def _int(v) -> int:
    try:
        return int(round(float(v)))
    except (TypeError, ValueError):
        return 0


def order_facts(row) -> list[str]:
    """Bullet lines describing an order line WITHOUT country or postage."""
    q, price = _int(row["quantity"]), float(row["price"])
    med = row.get("product_median_price")
    lines = [
        f"Product: {str(row.get('description', '')).title()} (category: {str(row['tag_category']).replace('_', ' ')}, "
        f"material: {row['tag_material']}, fragile: {'yes' if float(row['tag_fragile']) else 'no'}, "
        f"seasonal: {'yes' if float(row['tag_seasonal']) else 'no'})",
        f"Quantity {q} at GBP {price:.2f} each (line value GBP {q * price:.2f})"
        + (f"; usual price for this product GBP {float(med):.2f}" if med is not None and not pd.isna(med) and float(med) > 0 else ""),
        f"Basket: {_int(row['basket_lines'])} different products, total GBP {float(row['basket_value']):,.0f}",
    ]
    if _int(row["cust_prior_invoices"]) == 0:
        lines.append("Customer: first order from this customer (no purchase history)")
    else:
        lines.append(
            f"Customer history: {_int(row['cust_prior_invoices'])} earlier orders, {_int(row['cust_prior_lines'])} items "
            f"bought, {_int(row['cust_prior_returns'])} items returned before; customer for {_int(row['cust_tenure_days'])} "
            f"days, last order {_int(row['cust_recency_days'])} days ago")
    lines.append(f"Product history: sold {_int(row['prod_prior_lines'])} times before, returned "
                 f"{_int(row['prod_prior_returns'])} times")
    dt = pd.Timestamp(row["invoice_date"])
    lines.append(f"Order date: {dt:%d %b %Y}, {dt:%H:%M}")
    return lines


def country_facts(row, include_postage: bool, include_country: bool) -> list[str]:
    out = []
    if include_postage:
        out.append(f"Postage charged on this order: {'yes' if float(row['has_postage']) else 'no'}")
    if include_country:
        out.append(f"Customer's country: {row['country']}")
    return out


FACTOR_PHRASE = {
    "Customer history": lambda r: ("first order from this customer" if _int(r["cust_prior_invoices"]) == 0 else
                                   f"customer returned {_int(r['cust_prior_returns'])} of {_int(r['cust_prior_lines'])} "
                                   f"earlier items"),
    "Product history": lambda r: (f"product returned {_int(r['prod_prior_returns'])} times in "
                                  f"{_int(r['prod_prior_lines'])} earlier sales"),
    "Product attributes": lambda r: (("fragile " if float(r["tag_fragile"]) else "")
                                     + ("seasonal " if float(r["tag_seasonal"]) else "")
                                     + f"{str(r['tag_category']).replace('_', ' ')} item ({r['tag_material']})"),
    "Order & basket": lambda r: f"{_int(r['quantity'])} units, basket of {_int(r['basket_lines'])} products",
    "Timing": lambda r: f"ordered in {pd.Timestamp(r['invoice_date']):%B}",
}


def factor_phrases(row, factors: list[tuple[str, float]], k: int = 3) -> list[str]:
    """Readable evidence for the top factors; country-related groups are never described."""
    out = []
    for g, delta in factors:
        if g in FACTOR_PHRASE and abs(delta) > 1e-5:
            direction = "raises" if delta > 0 else "lowers"
            out.append(f"{FACTOR_PHRASE[g](row)} ({direction} risk)")
        if len(out) >= k:
            break
    return out


def pct(p: float) -> str:
    return "n/a" if p is None or (isinstance(p, float) and math.isnan(p)) else f"{100 * p:.1f}%"
