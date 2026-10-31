"""Deterministic verification of the reported Fair Returns metrics.

This script recomputes the mean and standard deviation of every reported
metric from seeds.csv and verifies that summary.csv matches those values.
It also checks that group_metrics.csv contains six country groups per method
and that the seed-0 demographic parity gap equals the reported seed-0 value.
The verification is fully offline and does not load a model checkpoint.
"""

from __future__ import annotations

import math
from pathlib import Path

import pandas as pd


DATA_DIR = Path(__file__).resolve().parents[1] / "deliverables" / "data"
SUMMARY_PATH = DATA_DIR / "summary.csv"
SEEDS_PATH = DATA_DIR / "seeds.csv"
GROUPS_PATH = DATA_DIR / "group_metrics.csv"

TARGET_METHODS = {"lr_aware", "mlp", "vib_grl", "vib_grl_dp"}
EXPECTED_GROUPS = {
    "United Kingdom",
    "Germany",
    "France",
    "EIRE",
    "Other Europe",
    "Rest of World",
}


def fmt(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.4f}"


def main() -> int:
    if not all(path.exists() for path in (SUMMARY_PATH, SEEDS_PATH, GROUPS_PATH)):
        print("Missing deterministic data files under deliverables/data.")
        return 2

    summary = pd.read_csv(SUMMARY_PATH)
    seeds = pd.read_csv(SEEDS_PATH)
    groups = pd.read_csv(GROUPS_PATH)
    failures: list[str] = []

    for method in sorted(TARGET_METHODS):
        summary_row = summary[summary["method"] == method]
        seed_rows = seeds[seeds["method"] == method]
        if summary_row.empty or seed_rows.empty:
            failures.append(f"{method}: missing summary or seed rows")
            continue

        summary_row = summary_row.iloc[0]
        for metric in ("auc", "brier", "dp_gap", "fpr_gap"):
            expected_mean = float(summary_row[f"{metric}_mean"])
            expected_std = float(summary_row[f"{metric}_std"])
            computed_mean = float(seed_rows[metric].mean())
            computed_std = float(seed_rows[metric].std(ddof=1))
            mean_ok = math.isclose(expected_mean, computed_mean, rel_tol=0.0, abs_tol=1e-9)
            std_ok = math.isclose(expected_std, computed_std, rel_tol=0.0, abs_tol=1e-9)
            print(
                f"{method:12s} {metric:8s} "
                f"mean={fmt(computed_mean)} std={fmt(computed_std)} "
                f"{'PASS' if (mean_ok and std_ok) else 'FAIL'}"
            )
            if not (mean_ok and std_ok):
                failures.append(f"{method}.{metric}: summary differs from seeds")

        method_groups = groups[groups["method"] == method]
        observed_groups = set(method_groups["group"].dropna().unique())
        if observed_groups != EXPECTED_GROUPS:
            failures.append(f"{method}: unexpected country groups {sorted(observed_groups)}")

        seed0 = seed_rows[seed_rows["seed"] == 0]
        if not seed0.empty:
            expected_dp = float(seed0.iloc[0]["dp_gap"])
            computed_dp = float(method_groups["flag_rate"].max() - method_groups["flag_rate"].min())
            dp_ok = math.isclose(expected_dp, computed_dp, rel_tol=0.0, abs_tol=1e-9)
            print(
                f"{method:12s} {'dp_seed0':8s} "
                f"expected={fmt(expected_dp)} computed={fmt(computed_dp)} "
                f"{'PASS' if dp_ok else 'FAIL'}"
            )
            if not dp_ok:
                failures.append(f"{method}.dp_seed0: group_metrics differs from seeds")

    if failures:
        print(f"\nVerification failed with {len(failures)} issue(s).")
        return 1

    print("\nAll deterministic checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
