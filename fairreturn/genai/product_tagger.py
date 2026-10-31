"""Product attributes from free-text descriptions: a keyword rule tagger and an LLM tagger.

The descriptions in Online Retail II are short upper-case titles ("WHITE HANGING HEART T-LIGHT HOLDER").
We extract four attributes that plausibly drive returns and use them as model features:
    category  (product type)      material (main material)
    fragile   (breaks in transit)  seasonal (Christmas / Easter / Valentine ...)

The keyword rules (rule_tag / tag_products_with_rules) are deterministic and always available; they are
also the baseline the LLM tags are evaluated against. tag_products_with_llm labels the most frequent products
in batches and falls back to the rules for any invalid or missing answer.

CLI (writes outputs/genai/product_tags_llm.csv and outputs/genai/tag_eval.json):
    python -m fairreturn.genai.product_tagger --backend mock --top 300      # offline pipeline test
    python -m fairreturn.genai.product_tagger --backend api  --top 1000     # real run (~40 calls)
"""
from __future__ import annotations

import argparse
import re

import numpy as np
import pandas as pd

from fairreturn import config as cfg
from fairreturn.genai.llm_client import LLMClient, extract_json

CATEGORIES = ["bags_storage", "kitchen_dining", "candles_lighting", "home_decor", "stationery_wrap",
              "jewellery_accessories", "toys_kids", "party_supplies", "garden_outdoor", "textiles_warmers", "other"]
MATERIALS = ["glass", "ceramic", "metal", "wood", "textile", "paper", "plastic", "wax", "mixed", "unknown"]

# Ordered: the first matching rule wins, so specific phrases come before generic words.
CATEGORY_RULES: list[tuple[str, list[str]]] = [
    ("textiles_warmers", ["HOT WATER BOTTLE", "HOTTIE", "HAND WARMER", "THROW", "BLANKET", "TOWEL", "QUILT",
                          "TEA COSY", "OVEN GLOVE", "CUSHION COVER", "SCARF", "GLOVES", "SLIPPERS"]),
    ("party_supplies", ["PAPER CHAIN", "BUNTING", "BALLOON", "PARTY", "CRACKER", "CONFETTI", "PINATA", "GARLAND"]),
    ("toys_kids", ["SNAP CARDS", "CARD GAME", "TOY", "GAME", "JIGSAW", "PUZZLE", "DOLL", "CHILDRENS", "CHILDS",
                   "KIDS", "PLAYHOUSE", "CRAYONS", "SKIPPING ROPE", "DOMINOES", "MARBLES", "TEDDY", "BUBBLES",
                   "SPINNING TOP", "COLOURING", "PLAY"]),
    ("kitchen_dining", ["LUNCH BOX", "CAKE CASES", "CAKESTAND", "CAKE STAND", "CAKE TIN", "BAKING", "TEA SET",
                        "COOKIE CUTTER", "EGG CUP", "RECIPE", "PANTRY", "MUG", "CUP", "SAUCER", "TEACUP", "PLATE",
                        "BOWL", "CUTLERY", "JUG", "TRAY", "NAPKIN", "NAPKINS", "COASTER", "KITCHEN", "SPOON",
                        "JAR", "APRON", "SCALES", "DISH", "TEAPOT", "CHOPSTICKS", "PLACEMAT", "BEAKER",
                        "TUMBLER", "WATER BOTTLE", "SNACK BOXES", "TINS", "STRAW", "STRAWS", "BOTTLE"]),
    ("bags_storage", ["LUNCH BAG", "SHOPPER", "CHARLOTTE BAG", "STORAGE", "BASKET", "HAMPER", "BAG", "BAGS",
                      "TOTE", "SACK", "CASE"]),
    ("candles_lighting", ["T-LIGHT", "TLIGHT", "T LIGHT", "CANDLE", "CANDLES", "LANTERN", "LAMPSHADE", "LAMP",
                          "NIGHT LIGHT", "LIGHTS", "LIGHT", "INCENSE", "OIL BURNER", "CANDLEHOLDER"]),
    ("stationery_wrap", ["GIFT TAG", "GIFT TAGS", "WRAPPING", "WRAP", "CARDS", "CARD", "NOTEBOOK", "NOTEBOOKS",
                         "NOTE", "PEN", "PENS", "PENCIL", "PENCILS", "STICKER", "STICKERS", "TAPE", "ALBUM",
                         "BOOK", "ENVELOPE", "ENVELOPES", "CALENDAR", "DIARY", "ERASER", "RULER", "TISSUE",
                         "TISSUES", "PAPER", "STATIONERY", "MEMO", "RIBBON", "RIBBONS"]),
    ("jewellery_accessories", ["NECKLACE", "BRACELET", "EARRINGS", "RING", "CHARM", "HAIR", "BEAD", "BEADS",
                               "DIAMANTE", "BROOCH", "PURSE", "WALLET", "UMBRELLA", "KEYRING", "KEY RING",
                               "SUNGLASSES", "HANDBAG", "BANGLE", "PENDANT", "JEWEL"]),
    ("garden_outdoor", ["GARDEN", "PLANT", "PLANTER", "WATERING", "BIRD FEEDER", "PICNIC", "PARASOL",
                        "OUTDOOR", "FLOWER POT", "SEED", "SEEDS", "TRUG", "WINDMILL"]),
    ("home_decor", ["SIGN", "FRAME", "MIRROR", "CLOCK", "DECORATION", "DECORATIONS", "ORNAMENT", "WREATH",
                    "DOORMAT", "CUSHION", "CABINET", "DRAWER", "HOOK", "HOOKS", "RACK", "DOORSTOP", "VASE",
                    "PICTURE", "CHALKBOARD", "BLACKBOARD", "BIRDHOUSE", "HEART", "TILE", "WALL", "SHELF",
                    "PLAQUE", "HANGER", "MAGNET", "MAGNETS", "TRINKET", "ART", "BOX", "BOXES", "HOLDER",
                    "BAUBLE", "STAR", "TREE", "POT", "DOOR", "KNOB", "BUILDING BLOCK"]),
]
MATERIAL_RULES: list[tuple[str, list[str]]] = [
    ("glass", ["GLASS", "GLASSES", "CRYSTAL"]),
    ("ceramic", ["CERAMIC", "PORCELAIN", "CHINA", "STONEWARE", "TERRACOTTA", "POTTERY"]),
    ("metal", ["METAL", "ZINC", "TIN", "TINS", "ENAMEL", "SILVER", "IRON", "WIRE", "ALUMINIUM", "CHROME",
               "STEEL", "BRASS", "COPPER"]),
    ("wood", ["WOOD", "WOODEN", "BAMBOO", "WICKER", "CORK", "RATTAN"]),
    ("textile", ["FELT", "COTTON", "KNITTED", "CROCHET", "FABRIC", "WOOL", "WOOLLY", "LACE", "CANVAS", "JUTE",
                 "VELVET", "SILK", "FLANNEL", "LINEN", "HESSIAN", "FLEECE", "CUSHION"]),
    ("paper", ["PAPER", "CARD", "CARDS", "CARDBOARD", "TISSUE", "NAPKINS", "NOTEBOOK", "WRAP", "ENVELOPE"]),
    ("plastic", ["PLASTIC", "ACRYLIC", "RUBBER", "VINYL", "MELAMINE", "PVC", "PERSPEX"]),
    ("wax", ["CANDLE", "CANDLES", "WAX"]),
]
FRAGILE_WORDS = ["GLASS", "CERAMIC", "PORCELAIN", "CHINA", "MIRROR", "CRYSTAL", "BAUBLE", "STONEWARE",
                 "TERRACOTTA", "POTTERY"]
SEASONAL_WORDS = ["CHRISTMAS", "XMAS", "EASTER", "HALLOWEEN", "VALENTINE", "ADVENT", "SANTA", "REINDEER",
                  "SNOWMAN", "SNOWFLAKE", "NATIVITY", "BAUBLE", "MOTHERS DAY", "FATHERS DAY"]


def _norm(desc: str) -> str:
    d = re.sub(r"[^A-Z0-9\- ]", " ", str(desc).upper())
    return " " + re.sub(r"\s+", " ", d).strip() + " "


def _singular(text: str) -> str:
    """Crude plural stripping so 'BOWLS' matches 'BOWL' (words ending in SS are kept)."""
    return " " + " ".join(w[:-1] if len(w) > 3 and w.endswith("S") and not w.endswith("SS") else w
                          for w in text.split()) + " "


def _has(text: str, phrase: str) -> bool:
    return f" {phrase} " in text


def rule_tag(description: str) -> dict:
    t = _norm(description)
    t = t + _singular(t)  # match both the original and the singular form
    category = next((c for c, words in CATEGORY_RULES if any(_has(t, w) for w in words)), "other")
    mats = [m for m, words in MATERIAL_RULES if any(_has(t, w) for w in words)]
    material = mats[0] if len(mats) == 1 else ("mixed" if len(mats) > 1 else "unknown")
    if len(mats) > 1 and "glass" in mats:
        material = "glass"  # the breakable part matters most for returns
    fragile = any(_has(t, w) for w in FRAGILE_WORDS) or material in ("glass", "ceramic")
    seasonal = any(_has(t, w) for w in SEASONAL_WORDS)
    return {"category": category, "material": material, "fragile": bool(fragile), "seasonal": bool(seasonal)}


def tag_products_with_rules(products: pd.DataFrame) -> pd.DataFrame:
    """products: columns stock_code, description -> adds tag_* columns (source='rules')."""
    tags = pd.DataFrame([rule_tag(d) for d in products["description"]], index=products.index)
    out = products[["stock_code", "description"]].copy()
    out["tag_category"], out["tag_material"] = tags["category"], tags["material"]
    out["tag_fragile"], out["tag_seasonal"] = tags["fragile"].astype(int), tags["seasonal"].astype(int)
    out["tag_source"] = "rules"
    return out


# ----------------------------------------------------------------------------- LLM tagger
SYSTEM_PROMPT = ("You label products sold by a UK online gift and homeware wholesaler. You answer with JSON only "
                 "and only use the allowed values.")


def build_tag_prompt(items: list[tuple[int, str]]) -> str:
    lines = "\n".join(f"{i}. {d}" for i, d in items)
    return (
        "For each product description, give:\n"
        f"- category: one of {CATEGORIES}\n"
        f"- material: the main material, one of {MATERIALS} (use 'unknown' if the text does not say)\n"
        "- fragile: true if the item is likely to break or chip in transit, else false\n"
        "- seasonal: true if the item is meant for a specific season or holiday (e.g. Christmas, Easter), else false\n\n"
        "Return a JSON array with one object per product, in the same order, like\n"
        '[{"id": 1, "category": "...", "material": "...", "fragile": false, "seasonal": false}]\n\n'
        f"Products:\n{lines}"
    )


def mock_tag_reply(system: str, user: str) -> str:
    """Offline stand-in: rule tags with deterministic disagreements so the evaluation code is exercised."""
    import json
    out = []
    for line in user.split("Products:\n", 1)[1].splitlines():
        m = re.match(r"(\d+)\. (.*)", line.strip())
        if not m:
            continue
        i, desc = int(m.group(1)), m.group(2)
        tag = rule_tag(desc)
        h = sum(ord(ch) for ch in desc)
        if h % 9 == 0:
            tag["category"] = CATEGORIES[h % len(CATEGORIES)]
        if h % 13 == 0:
            tag["material"] = "mixed"
        if h % 31 == 0:
            tag["category"] = "decorative_item"   # an invalid value, to test validation
        out.append({"id": i, **tag})
    return json.dumps(out)


def _validate(obj: dict) -> tuple[dict | None, str]:
    try:
        cat = str(obj.get("category", "")).strip().lower()
        mat = str(obj.get("material", "")).strip().lower()
        fragile = obj.get("fragile")
        seasonal = obj.get("seasonal")
        if isinstance(fragile, str):
            fragile = fragile.strip().lower() in ("true", "yes", "1")
        if isinstance(seasonal, str):
            seasonal = seasonal.strip().lower() in ("true", "yes", "1")
    except Exception as e:  # noqa: BLE001
        return None, f"bad object: {e}"
    problems = []
    if cat not in CATEGORIES:
        problems.append(f"category={cat!r}")
    if mat not in MATERIALS:
        problems.append(f"material={mat!r}")
    if not isinstance(fragile, bool) or not isinstance(seasonal, bool):
        problems.append("non-boolean flag")
    if problems:
        return None, "; ".join(problems)
    return {"category": cat, "material": mat, "fragile": fragile, "seasonal": seasonal}, ""


def tag_products_with_llm(products: pd.DataFrame, client: LLMClient, batch_size: int = 25,
                          workers: int = 4) -> pd.DataFrame:
    """Returns one row per product with LLM tags (or rule fallback) and validation info."""
    items = products[["stock_code", "description"]].reset_index(drop=True)
    requests_ = []
    for start in range(0, len(items), batch_size):
        chunk = items.iloc[start:start + batch_size]
        numbered = [(k + 1, d) for k, d in enumerate(chunk["description"])]
        requests_.append({"system": SYSTEM_PROMPT, "user": build_tag_prompt(numbered), "mock_fn": mock_tag_reply,
                          "meta": {"task": "product_tags", "start": int(start)}})
    print(f"  tagging {len(items)} products in {len(requests_)} LLM calls ({client.backend}:{client.model})")
    replies = client.chat_many(requests_, workers=workers, progress_every=10)

    rows = []
    for b, reply in enumerate(replies):
        chunk = items.iloc[b * batch_size:(b + 1) * batch_size]
        try:
            parsed = extract_json(reply, prefer="array")
            if isinstance(parsed, dict):
                parsed = next((v for v in parsed.values() if isinstance(v, list)), [])
        except ValueError:
            parsed = []
        by_id = {}
        for obj in parsed if isinstance(parsed, list) else []:
            if isinstance(obj, dict) and "id" in obj:
                try:
                    by_id[int(obj["id"])] = obj
                except (TypeError, ValueError):
                    continue
        for k, (_, prod) in enumerate(chunk.iterrows()):
            obj = by_id.get(k + 1)
            tag, problem = (None, "missing in reply") if obj is None else _validate(obj)
            fallback = rule_tag(prod["description"])
            final = tag or fallback
            rows.append({
                "stock_code": prod["stock_code"], "description": prod["description"],
                "tag_category": final["category"], "tag_material": final["material"],
                "tag_fragile": int(final["fragile"]), "tag_seasonal": int(final["seasonal"]),
                "tag_source": ("llm" if tag else "rules_fallback") + ("_mock" if client.is_mock else ""),
                "llm_valid": int(tag is not None), "llm_problem": problem,
                "llm_raw_category": "" if obj is None else str(obj.get("category", "")),
            })
    return pd.DataFrame(rows)


def main():
    from fairreturn.genai import tag_eval
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backend", choices=["auto", "api", "mock"], default="auto")
    ap.add_argument("--top", type=int, default=1000, help="tag the N products with most purchase lines")
    ap.add_argument("--batch-size", type=int, default=25)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    if not cfg.PRODUCTS_FILE.exists():
        raise SystemExit("Run `python -m fairreturn.data.prepare` first (it writes data/processed/products.csv).")
    products = pd.read_csv(cfg.PRODUCTS_FILE, dtype={"stock_code": str}, keep_default_na=False)
    top = products.sort_values("n_lines", ascending=False).head(a.top)
    cfg.GENAI_DIR.mkdir(parents=True, exist_ok=True)
    client = LLMClient(backend=a.backend, cache_file=cfg.GENAI_DIR / "llm_cache_tags.jsonl")
    tags = tag_products_with_llm(top, client, batch_size=a.batch_size, workers=a.workers)
    tags.to_csv(cfg.LLM_TAGS_FILE, index=False)
    print(f"  wrote {cfg.LLM_TAGS_FILE} ({len(tags)} products, valid LLM answers: {tags['llm_valid'].mean():.1%})")
    tag_eval.ensure_human_template(products)
    report = tag_eval.evaluate(products, tags, client.describe())
    print(tag_eval.format_report(report))


if __name__ == "__main__":
    main()
