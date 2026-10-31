"""Accuracy, group fairness, counterfactual and leakage metrics.

Group metrics are computed at a single decision threshold: an order line is flagged 'high return risk'
if its score is in the top HIGH_RISK_SHARE of validation scores. Equalized-odds difference follows the
fairlearn definition: max(TPR gap, FPR gap) across groups.

Order lines of one customer are strongly correlated (EIRE, for example, is essentially three wholesale
customers), so a group only enters the TPR gap if its returns come from at least MIN_CUSTOMERS distinct
customers, and the FPR gap if its non-returned lines do. Excluded groups are listed in the output.
The FPR gap is the most direct measure of the harm we care about: customers who did NOT return
anything being flagged as high-risk at different rates depending on where they live.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, balanced_accuracy_score, brier_score_loss, roc_auc_score
from sklearn.neural_network import MLPClassifier

from fairreturn import config as cfg


def accuracy_metrics(y, p) -> dict:
    y, p = np.asarray(y).astype(int), np.asarray(p, dtype=float)
    out = {"auc": float("nan"), "pr_auc": float("nan"), "brier": float(brier_score_loss(y, np.clip(p, 0, 1)))}
    if len(np.unique(y)) > 1:
        out["auc"] = float(roc_auc_score(y, p))
        out["pr_auc"] = float(average_precision_score(y, p))
    out["base_rate"] = float(y.mean())
    out["mean_score"] = float(p.mean())
    return out


def threshold_top_share(p_val, share: float = cfg.HIGH_RISK_SHARE) -> float:
    return float(np.quantile(np.asarray(p_val, dtype=float), 1 - share))


MIN_CUSTOMERS = 10


def group_table(y, p, groups, threshold: float, customers=None) -> pd.DataFrame:
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=float)
    groups = np.asarray(groups)
    cust = np.asarray(customers) if customers is not None else None
    flag = p >= threshold
    rows = []
    for g in cfg.GROUPS:
        m = groups == g
        if not m.any():
            continue
        yy, ff, pp = y[m], flag[m], p[m]
        pos, neg = yy == 1, yy == 0
        cc = cust[m] if cust is not None else None
        rows.append({
            "group": g, "n": int(m.sum()), "positives": int(pos.sum()),
            "customers": int(len(np.unique(cc))) if cc is not None else None,
            "pos_customers": int(len(np.unique(cc[pos]))) if cc is not None else None,
            "neg_customers": int(len(np.unique(cc[neg]))) if cc is not None else None,
            "flag_rate": float(ff.mean()),
            "tpr": float(ff[pos].mean()) if pos.any() else np.nan,
            "fpr": float(ff[neg].mean()) if neg.any() else np.nan,
            "mean_score": float(pp.mean()), "observed_rate": float(yy.mean()),
            "calibration_gap": float(pp.mean() - yy.mean()),
        })
    return pd.DataFrame(rows)


def fairness_gaps(gt: pd.DataFrame, min_pos: int = 30, min_neg: int = 300,
                  min_customers: int = MIN_CUSTOMERS) -> dict:
    ok_t = gt[gt["positives"] >= min_pos]
    ok_f = gt[(gt["n"] - gt["positives"]) >= min_neg]
    if "pos_customers" in gt and gt["pos_customers"].notna().all():
        ok_t = ok_t[ok_t["pos_customers"] >= min_customers]
        ok_f = ok_f[ok_f["neg_customers"] >= min_customers]
    tpr_gap = float(ok_t["tpr"].max() - ok_t["tpr"].min()) if len(ok_t) > 1 else float("nan")
    fpr_gap = float(ok_f["fpr"].max() - ok_f["fpr"].min()) if len(ok_f) > 1 else float("nan")
    return {
        "eo_gap": float(np.nanmax([tpr_gap, fpr_gap])), "tpr_gap": tpr_gap, "fpr_gap": fpr_gap,
        "dp_gap": float(gt["flag_rate"].max() - gt["flag_rate"].min()),
        "calibration_gap_max": float(gt["calibration_gap"].abs().max()),
        "mean_score_ratio_max_min": float(gt["mean_score"].max() / max(1e-9, gt["mean_score"].min())),
        "groups_in_tpr_gap": ok_t["group"].tolist(), "groups_in_fpr_gap": ok_f["group"].tolist(),
    }


def counterfactual_metrics(p, p_twin, threshold: float) -> dict:
    p, pt = np.asarray(p, dtype=float), np.asarray(p_twin, dtype=float)
    flip = (p >= threshold) != (pt >= threshold)
    rel = np.abs(pt - p) / np.maximum(p, 1e-6)
    return {"cf_flip_rate": float(flip.mean()), "cf_mean_abs_delta": float(np.mean(np.abs(pt - p))),
            "cf_median_rel_change": float(np.median(rel)),
            "cf_flip_rate_flagged": float(flip[p >= threshold].mean()) if (p >= threshold).any() else float("nan")}


def _balanced_sample(a, per_group: int, rng):
    idx = []
    for g in np.unique(a):
        gi = np.flatnonzero(a == g)
        idx.append(rng.choice(gi, size=min(per_group, len(gi)), replace=False))
    return np.concatenate(idx)


def probe_representation(z_train, a_train, z_test, a_test, seed: int = cfg.SEED,
                         per_group: int = 3000, max_test: int = cfg.PROBE_MAX_TEST) -> dict:
    """Train FRESH classifiers on frozen representations to recover the country group.

    Never use the training-time adversary for this - it can be fooled while the information remains
    (Elazar & Goldberg, 2018). Balanced accuracy; chance level = 1 / number of groups.
    """
    rng = np.random.default_rng(seed)
    a_train, a_test = np.asarray(a_train), np.asarray(a_test)
    tr = _balanced_sample(a_train, per_group, rng)
    te = np.arange(len(a_test))
    if len(te) > max_test:
        te = _balanced_sample(a_test, max_test // len(np.unique(a_test)), rng)
    zt, at = z_train[tr], a_train[tr]
    mu, sd = zt.mean(0), zt.std(0) + 1e-6
    zt, zv = (zt - mu) / sd, (z_test[te] - mu) / sd
    out = {"chance": 1.0 / len(np.unique(a_train))}
    lr = LogisticRegression(max_iter=3000).fit(zt, at)
    out["probe_lr"] = float(balanced_accuracy_score(a_test[te], lr.predict(zv)))
    mlp = MLPClassifier(hidden_layer_sizes=(64,), max_iter=400, random_state=seed, early_stopping=True).fit(zt, at)
    out["probe_mlp"] = float(balanced_accuracy_score(a_test[te], mlp.predict(zv)))
    out["probe_best"] = max(out["probe_lr"], out["probe_mlp"])
    return out


def probe_output(p_train, y_train, a_train, p_test, y_test, a_test, seed: int = cfg.SEED) -> dict:
    """How much country information is in the risk score itself (given the true label)?

    This is exactly what an Equalized-Odds adversary looks at: (score, label) -> country group.
    """
    def feats(p, y):
        lp = np.log(np.clip(p, 1e-6, 1)) - np.log(np.clip(1 - p, 1e-6, 1))
        y = np.asarray(y, dtype=float)
        return np.column_stack([lp, y, lp * y])
    rng = np.random.default_rng(seed)
    a_train = np.asarray(a_train)
    tr = _balanced_sample(a_train, 5000, rng)
    clf = LogisticRegression(max_iter=3000).fit(feats(p_train[tr], y_train[tr]), a_train[tr])
    pred = clf.predict(feats(p_test, y_test))
    return {"output_probe": float(balanced_accuracy_score(a_test, pred))}
