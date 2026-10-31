"""Project-wide settings: paths, country groups, split dates and modelling defaults.

Everything the pipeline, the experiments and the web app need to agree on lives here, so a change
(for example a different return window) only has to be made once.
"""
from __future__ import annotations

from pathlib import Path

# ----------------------------------------------------------------------------- paths
ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"            # put online_retail_1.xlsx / online_retail_2.xlsx (or the Kaggle CSV) here
DATA_INTERIM = ROOT / "data" / "interim"    # cached, cleaned transactions
DATA_PROCESSED = ROOT / "data" / "processed"
OUTPUTS = ROOT / "outputs"
MODELS_DIR = OUTPUTS / "models"
RESULTS_DIR = OUTPUTS / "results"
PREDS_DIR = RESULTS_DIR / "preds"
FIG_DIR = OUTPUTS / "figures"
GENAI_DIR = OUTPUTS / "genai"
AUDIT_DIR = GENAI_DIR / "audit"
DEMO_DIR = OUTPUTS / "demo"
FRONTEND_DIR = ROOT / "frontend"

LINES_FILE = DATA_PROCESSED / "lines.pkl"
TWINS_FILE = DATA_PROCESSED / "twins_test.pkl"
META_FILE = DATA_PROCESSED / "meta.json"
PRODUCTS_FILE = DATA_PROCESSED / "products.csv"
CUSTOMER_SNAPSHOT_FILE = DATA_PROCESSED / "customers_snapshot.csv"
RULE_TAGS_FILE = DATA_PROCESSED / "product_tags_rules.csv"
LLM_TAGS_FILE = GENAI_DIR / "product_tags_llm.csv"
HUMAN_TAG_TEMPLATE = DATA_PROCESSED / "human_tag_sample.csv"

# ----------------------------------------------------------------------------- data rules
# Stock codes that are not products (postage, fees, manual adjustments, vouchers ...).
SERVICE_CODES = {
    "POST", "DOT", "C2", "M", "D", "S", "B", "BANK CHARGES", "ADJUST", "ADJUST2",
    "AMAZONFEE", "CRUK", "TEST001", "TEST002", "PADS",
}
SERVICE_PREFIXES = ("GIFT_",)
# Shipping charges that are set by the destination: a strong proxy for the customer's country.
POSTAGE_CODES = {"POST", "C2"}

# Sensitive attribute: the customer's country of residence, grouped so every group has enough data.
GROUPS = ["United Kingdom", "Germany", "France", "EIRE", "Other Europe", "Rest of World"]
OTHER_EUROPE = {
    "Netherlands", "Spain", "Belgium", "Portugal", "Italy", "Sweden", "Finland", "Austria",
    "Denmark", "Greece", "Cyprus", "Poland", "Lithuania", "Malta", "Czech Republic",
    "European Community", "Norway", "Switzerland", "Iceland", "Channel Islands",
}
GROUP_SHORT = {"United Kingdom": "UK", "Germany": "DE", "France": "FR", "EIRE": "IE",
               "Other Europe": "EU-other", "Rest of World": "RoW"}


def country_group(country: str) -> str:
    """Map a raw country name to one of GROUPS."""
    c = str(country).strip()
    if c in ("United Kingdom", "Germany", "France", "EIRE"):
        return c
    if c in OTHER_EUROPE:
        return "Other Europe"
    return "Rest of World"


# ----------------------------------------------------------------------------- labels and splits
HORIZON_DAYS = 60            # a purchase line is "returned" if a matching C-invoice line follows within 60 days
SPLIT_DATES = {
    "train_start": "2010-03-01",   # Dec 2009 - Feb 2010 is burn-in: used only to build customer/product history
    "val_start": "2011-03-01",
    "test_start": "2011-06-01",    # test ends HORIZON_DAYS before the last date in the data (label window must close)
}
SMOOTH_ALPHA_CUSTOMER = 10.0   # Bayesian smoothing for history return rates (pseudo-count)
SMOOTH_ALPHA_PRODUCT = 20.0

# ----------------------------------------------------------------------------- modelling defaults
HIGH_RISK_SHARE = 0.10        # top 10 % of validation scores = "high return risk" flag
SEED = 0
PROBE_MAX_TRAIN = 20000
PROBE_MAX_TEST = 20000

# Models the web app loads (must exist in outputs/models after experiments.run_all); the first one is the
# default comparison on the order-check page.
DEMO_MODELS = ["gbdt_aware", "lr_aware", "gbdt_unaware", "gbdt_blind", "gbdt_reweigh", "vib_grl_proxy", "vib_grl"]
DEMO_BASELINE = "gbdt_aware"
DEMO_OURS = "vib_grl"
