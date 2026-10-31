"""If PyTorch is installed: the torch port must reproduce the NumPy model's losses and gradients exactly."""
from __future__ import annotations

import numpy as np

from fairreturn.models.vib_grl import ADV_PARAMS, MAIN_PARAMS, FairReturnModel


def test_torch_matches_numpy():
    try:
        import torch
    except ImportError:
        return "skipped (torch not installed)"
    from fairreturn.models.vib_grl_torch import TorchFairReturnModel
    kw = dict(z_dim=4, hidden=(6, 5), head_hidden=4, adv_hidden=5, beta=0.05, lambda_max=1.0,
              condition_on_y=True, pos_weight=2.0, seed=0, dtype="float64")
    npm = FairReturnModel(5, 3, **kw)
    tm = TorchFairReturnModel(5, 3, **kw, device="cpu")
    tm.load_numpy_params(npm.params)
    gw = np.array([0.5, 1.5, 2.0])
    npm.group_weights = gw
    tm.group_weights = torch.as_tensor(gw, dtype=torch.float64)
    rng = np.random.default_rng(1)
    x, y, a = rng.standard_normal((7, 5)), (rng.random(7) < 0.4).astype(float), rng.integers(0, 3, 7)
    eps = rng.standard_normal((7, 4))
    lam = 0.7
    losses, g, ga, _ = npm.forward_backward(x, y, a, lam, eps=eps)
    X, Y, A = torch.as_tensor(x), torch.as_tensor(y), torch.as_tensor(a)
    ly, kl, la = tm.losses(X, Y, A, lam, eps=torch.as_tensor(eps))
    assert abs(float(ly) - losses["loss_y"]) < 1e-9 and abs(float(la) - losses["loss_adv"]) < 1e-9
    tm.net.zero_grad()
    (ly + kw["beta"] * kl + la).backward()
    for name in TorchFairReturnModel.MAP:
        layer = getattr(tm.net, name)
        ref = (ga if f"{name}_W" in ADV_PARAMS else g)
        if f"{name}_W" not in ref:
            continue
        assert np.allclose(layer.weight.grad.numpy().T, ref[f"{name}_W"], atol=1e-8), name
        assert np.allclose(layer.bias.grad.numpy(), ref[f"{name}_b"], atol=1e-8), name
    assert set(MAIN_PARAMS) | set(ADV_PARAMS)
