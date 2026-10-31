"""Fairness metrics, Reweighing and explanations on small hand-made examples with known answers."""
from __future__ import annotations

import numpy as np

from fairreturn import config as cfg
from fairreturn.evaluation import metrics as M
from fairreturn.models.baselines import reweighing_weights


def _two_groups(n_cust_de: int = 20):
    """UK and Germany, 1,000 lines each: returned lines are flagged 50% of the time in both groups, non-returned
    lines 10% (UK) vs 30% (Germany), so the true gaps are TPR 0 and FPR 0.2."""
    rng = np.random.default_rng(0)
    rows = []
    for g, fpr, n_cust in (("United Kingdom", 0.10, 50), ("Germany", 0.30, n_cust_de)):
        y = np.r_[np.ones(200), np.zeros(800)].astype(int)
        flag = np.r_[np.arange(200) < 100, np.arange(800) < int(800 * fpr)]
        p = np.where(flag, 0.9, 0.1)
        cust = rng.integers(0, n_cust, size=1000).astype(str)
        cust = np.char.add(g[:2] + "-", cust)
        rows.append((y, p, np.full(1000, g), cust))
    y, p, g, c = (np.concatenate(z) for z in zip(*rows))
    return y, p, g, c


def test_group_table_and_gaps():
    y, p, g, c = _two_groups()
    gt = M.group_table(y, p, g, threshold=0.5, customers=c)
    t = gt.set_index("group")
    assert abs(t.loc["United Kingdom", "fpr"] - 0.10) < 1e-9 and abs(t.loc["Germany", "fpr"] - 0.30) < 1e-9
    assert abs(t.loc["Germany", "tpr"] - 0.5) < 1e-9
    gaps = M.fairness_gaps(gt, min_pos=30, min_neg=300)
    assert abs(gaps["fpr_gap"] - 0.2) < 1e-9 and abs(gaps["tpr_gap"]) < 1e-9
    assert abs(gaps["eo_gap"] - 0.2) < 1e-9          # equalized-odds gap = max(TPR gap, FPR gap)


def test_small_groups_are_left_out_of_the_gaps():
    y, p, g, c = _two_groups(n_cust_de=3)            # Germany now has only three customers (like EIRE)
    gaps = M.fairness_gaps(M.group_table(y, p, g, 0.5, c), min_pos=30, min_neg=300)
    assert "Germany" not in gaps["groups_in_fpr_gap"] and np.isnan(gaps["fpr_gap"])


def test_counterfactual_flip_rate():
    p = np.array([0.2, 0.6, 0.7, 0.1])
    twin = np.array([0.2, 0.4, 0.8, 0.6])           # lines 2 and 4 cross the 0.5 threshold
    cf = M.counterfactual_metrics(p, twin, 0.5)
    assert abs(cf["cf_flip_rate"] - 0.5) < 1e-12
    assert abs(cf["cf_flip_rate_flagged"] - 0.5) < 1e-12


def test_threshold_flags_the_top_share():
    p = np.linspace(0, 1, 1001)
    thr = M.threshold_top_share(p, 0.1)
    assert abs((p >= thr).mean() - 0.1) < 0.002


def test_reweighing_makes_group_and_label_independent():
    rng = np.random.default_rng(1)
    a = rng.integers(0, 3, 5000)
    y = (rng.random(5000) < np.array([0.02, 0.05, 0.10])[a]).astype(int)   # base rates differ by group
    w = reweighing_weights(y, a)
    for g in range(3):
        rate = np.sum(w[(a == g) & (y == 1)]) / np.sum(w[a == g])
        assert abs(rate - y.mean()) < 1e-9          # after weighting every group has the overall rate


def test_explanations_recover_a_linear_model():
    from fairreturn.data.features import FeatureBuilder
    from fairreturn.evaluation.explain import contributions
    fb = FeatureBuilder("blind")
    fb.numeric = ["log_qty", "cust_prior_return_rate"]
    fb.categorical = []
    w = np.array([0.5, -2.0])
    predict = lambda X: X @ w  # noqa: E731 - a linear "model": occlusion must give exactly w * (x - reference)
    X = np.array([[1.0, 0.2], [3.0, -1.0]])
    ref = np.array([0.5, 0.0])
    out = contributions(predict, X, fb, ref)
    assert abs(out[1]["Order & basket"] - 0.5 * (3.0 - 0.5)) < 1e-12
    assert abs(out[1]["Customer history"] - (-2.0) * (-1.0 - 0.0)) < 1e-12
    assert set(out[0]) == {"Order & basket", "Customer history"}   # no postage / country group for 'blind'
    assert cfg.HIGH_RISK_SHARE == 0.10
