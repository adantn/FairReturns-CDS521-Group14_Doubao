"""Correctness tests for the NumPy VIB + GRL model (gradient check, GRL sign, persistence, learning)."""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

from fairreturn.models.vib_grl import ADV_PARAMS, MAIN_PARAMS, FairReturnModel


def _tiny(beta=0.05, lam_max=1.0, cond=True, seed=3):
    m = FairReturnModel(input_dim=5, n_groups=3, z_dim=4, hidden=(6, 5), head_hidden=4, adv_hidden=5,
                        beta=beta, lambda_max=lam_max, condition_on_y=cond, seed=seed, dtype="float64",
                        pos_weight=2.0)
    m.group_weights = np.array([0.5, 1.5, 2.0])
    rng = np.random.default_rng(seed)
    for k in m.params:  # non-zero biases keep every ReLU away from its kink (finite differences need that)
        if k.endswith("_b"):
            m.params[k] = m.params[k] + rng.normal(0, 0.3, m.params[k].shape)
    x = rng.standard_normal((7, 5))
    y = (rng.random(7) < 0.4).astype(float)
    a = rng.integers(0, 3, 7)
    eps = rng.standard_normal((7, 4))
    return m, x, y, a, eps


def _numeric_grad(fn, arr, h=1e-6):
    g = np.zeros_like(arr)
    it = np.nditer(arr, flags=["multi_index"])
    for _ in it:
        i = it.multi_index
        old = arr[i]
        arr[i] = old + h
        fp = fn()
        arr[i] = old - h
        fm = fn()
        arr[i] = old
        g[i] = (fp - fm) / (2 * h)
    return g


def _check(beta, lam_max, cond, lam=0.7):
    m, x, y, a, eps = _tiny(beta, lam_max, cond)
    losses, g, ga, _ = m.forward_backward(x, y, a, lam, eps=eps)

    def objective():
        return m.forward_backward(x, y, a, lam, eps=eps)[0]["objective"]

    def adv_loss():
        return m.forward_backward(x, y, a, lam, eps=eps)[0]["loss_adv"]

    worst = 0.0
    for k in MAIN_PARAMS:
        if k not in g:
            continue
        num = _numeric_grad(objective, m.params[k])
        err = np.max(np.abs(num - g[k])) / (np.max(np.abs(num)) + np.max(np.abs(g[k])) + 1e-12)
        worst = max(worst, err)
        assert err < 1e-5, f"main grad mismatch for {k}: rel err {err:.2e}"
    if lam_max > 0:
        for k in ADV_PARAMS:
            num = _numeric_grad(adv_loss, m.params[k])
            err = np.max(np.abs(num - ga[k])) / (np.max(np.abs(num)) + np.max(np.abs(ga[k])) + 1e-12)
            worst = max(worst, err)
            assert err < 1e-5, f"adversary grad mismatch for {k}: rel err {err:.2e}"
    return worst


def test_gradients_vib_grl_eo():
    _check(beta=0.05, lam_max=1.0, cond=True)


def test_gradients_vib_grl_dp():
    _check(beta=0.05, lam_max=1.0, cond=False)


def test_gradients_plain_mlp():
    _check(beta=0.0, lam_max=0.0, cond=True)


def test_gradient_reversal_sign():
    """The encoder gradient must equal grad(L_y + beta*KL) - lam * grad(L_adv)."""
    m, x, y, a, eps = _tiny()
    _, g_with, _, _ = m.forward_backward(x, y, a, 0.8, eps=eps)
    _, g_without, _, _ = m.forward_backward(x, y, a, 0.8, eps=eps, with_adversary=False)

    def adv_loss():
        return m.forward_backward(x, y, a, 0.8, eps=eps)[0]["loss_adv"]

    num_adv = _numeric_grad(adv_loss, m.params["enc1_W"])
    assert np.allclose(g_with["enc1_W"], g_without["enc1_W"] - 0.8 * num_adv, atol=1e-6)


def test_save_load_roundtrip():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((400, 8)).astype(np.float32)
    y = (X[:, 0] + rng.standard_normal(400) * 0.5 > 0.8).astype(float)
    a = (X[:, 1] > 0).astype(int)
    m = FairReturnModel(8, 2, z_dim=4, hidden=(16, 8), epochs=2, batch_size=64, seed=1)
    m.fit(X, y, a, verbose=False)
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "m.npz"
        m.save(p)
        m2 = FairReturnModel.load(p)
    assert np.allclose(m.predict_proba(X), m2.predict_proba(X), atol=1e-6)
    assert m2.history["epoch"] == [1, 2]


def test_learns_signal_and_adversary_reduces_leakage():
    """On toy data where the group leaks through a proxy, GRL should cut probe accuracy."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import balanced_accuracy_score, roc_auc_score
    rng = np.random.default_rng(0)
    n = 6000
    a = rng.integers(0, 2, n)
    signal = rng.standard_normal(n)
    proxy = a + 0.3 * rng.standard_normal(n)
    X = np.column_stack([signal, proxy, rng.standard_normal((n, 3))]).astype(np.float32)
    y = (rng.random(n) < 1 / (1 + np.exp(-(1.5 * signal + 1.2 * a - 1)))).astype(float)
    tr, te = slice(0, 4000), slice(4000, n)
    res = {}
    for name, lam in (("plain", 0.0), ("grl", 2.0)):
        m = FairReturnModel(5, 2, z_dim=4, hidden=(16, 8), beta=0.0, lambda_max=lam, condition_on_y=False,
                            epochs=25, batch_size=128, seed=0)
        m.fit(X[tr], y[tr], a[tr], verbose=False)
        z_tr, z_te = m.encode(X[tr]), m.encode(X[te])
        probe = LogisticRegression(max_iter=2000).fit(z_tr, a[tr])
        res[name] = (roc_auc_score(y[te], m.predict_proba(X[te])),
                     balanced_accuracy_score(a[te], probe.predict(z_te)))
    assert res["plain"][0] > 0.7, res
    assert res["grl"][1] < res["plain"][1] - 0.05, res   # leakage drops ...
    assert res["grl"][1] < 0.6, res                       # ... to near chance (0.5)
