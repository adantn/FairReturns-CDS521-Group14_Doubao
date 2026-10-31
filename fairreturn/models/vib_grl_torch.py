"""Optional PyTorch port of models/vib_grl.py (same architecture, loss and interface).

The project runs entirely on the NumPy implementation; use this port if you want a GPU, autograd, or to
extend the architecture. tests/test_torch_parity.py loads identical weights into both versions and checks
that losses and gradients match, so the two can be swapped safely.

    pip install torch
    python -m tests.selftest            # runs the parity test when torch is installed
"""
from __future__ import annotations

import math

import numpy as np

try:
    import torch
    from torch import nn
    import torch.nn.functional as F
except ImportError as e:  # pragma: no cover - torch is optional
    raise ImportError("vib_grl_torch needs PyTorch: pip install torch") from e


class _GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambd * grad_output, None


def grad_reverse(x, lambd: float):
    return _GradReverse.apply(x, lambd)


class _Net(nn.Module):
    def __init__(self, input_dim, n_groups, z_dim, hidden, head_hidden, adv_hidden, condition_on_y):
        super().__init__()
        self.enc1, self.enc2 = nn.Linear(input_dim, hidden[0]), nn.Linear(hidden[0], hidden[1])
        self.mu, self.lv = nn.Linear(hidden[1], z_dim), nn.Linear(hidden[1], z_dim)
        self.head1, self.head2 = nn.Linear(z_dim, head_hidden), nn.Linear(head_hidden, 1)
        self.adv1 = nn.Linear(z_dim + (1 if condition_on_y else 0), adv_hidden)
        self.adv2 = nn.Linear(adv_hidden, n_groups)
        self.condition_on_y = condition_on_y

    def encode(self, x):
        h = F.relu(self.enc2(F.relu(self.enc1(x))))
        return h, self.mu(h)

    def head(self, z):
        return self.head2(F.relu(self.head1(z))).squeeze(-1)

    def adversary(self, z, y):
        u = torch.cat([z, y.unsqueeze(1)], dim=1) if self.condition_on_y else z
        return self.adv2(F.relu(self.adv1(u)))


class TorchFairReturnModel:
    MAP = {"enc1": "enc1", "enc2": "enc2", "mu": "mu", "lv": "lv", "head1": "head1", "head2": "head2",
           "adv1": "adv1", "adv2": "adv2"}

    def __init__(self, input_dim, n_groups, z_dim=16, hidden=(64, 32), head_hidden=16, adv_hidden=32, beta=1e-3,
                 lambda_max=1.0, condition_on_y=True, lr=1e-3, adv_lr=None, adv_steps=3, batch_size=1024, epochs=15,
                 pos_weight=1.0, balanced_adversary=True, max_grad_norm=5.0, warmup=0.1, weight_decay=0.0,
                 select_epoch="last", seed=0, dtype="float32", device=None):
        self.config = dict(input_dim=input_dim, n_groups=n_groups, z_dim=z_dim, hidden=list(hidden),
                           head_hidden=head_hidden, adv_hidden=adv_hidden, beta=beta, lambda_max=lambda_max,
                           condition_on_y=condition_on_y, lr=lr, adv_lr=adv_lr or lr, adv_steps=adv_steps,
                           batch_size=batch_size, epochs=epochs, pos_weight=pos_weight,
                           balanced_adversary=balanced_adversary, max_grad_norm=max_grad_norm, warmup=warmup,
                           weight_decay=weight_decay, select_epoch=select_epoch, seed=seed, dtype=dtype)
        torch.manual_seed(seed)
        self.dtype = torch.float64 if dtype == "float64" else torch.float32
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.net = _Net(input_dim, n_groups, z_dim, hidden, head_hidden, adv_hidden, condition_on_y)
        self.net.to(self.device, self.dtype)
        with torch.no_grad():
            self.net.lv.weight.mul_(0.1)
            self.net.lv.bias.fill_(-2.0)
        self.group_weights = torch.ones(n_groups, dtype=self.dtype, device=self.device)
        self.history: dict = {}

    # weights shared with the NumPy model (W stored as in x @ W, i.e. transposed vs nn.Linear)
    def load_numpy_params(self, params: dict):
        with torch.no_grad():
            for name in self.MAP:
                layer = getattr(self.net, name)
                layer.weight.copy_(torch.as_tensor(params[f"{name}_W"].T, dtype=self.dtype))
                layer.bias.copy_(torch.as_tensor(params[f"{name}_b"], dtype=self.dtype))

    def lambda_at(self, progress: float) -> float:
        w = self.config["warmup"]
        if progress < w:
            return 0.0
        q = (progress - w) / max(1e-9, 1.0 - w)
        return self.config["lambda_max"] * (2.0 / (1.0 + math.exp(-10.0 * q)) - 1.0)

    def losses(self, x, y, a, lam, eps=None):
        c = self.config
        h, mu = self.net.encode(x)
        if c["beta"] > 0:
            lv = torch.clamp(self.net.lv(h), -8.0, 8.0)
            eps = torch.randn_like(mu) if eps is None else eps
            z = mu + torch.exp(0.5 * lv) * eps
            kl = (0.5 * (mu ** 2 + torch.exp(lv) - 1.0 - lv)).sum() / x.shape[0]
        else:
            z, kl = mu, torch.zeros((), dtype=self.dtype, device=self.device)
        self._last_z = z.detach()
        logit = self.net.head(z)
        w = torch.where(y > 0.5, torch.full_like(y, c["pos_weight"]), torch.ones_like(y))
        loss_y = (w * F.binary_cross_entropy_with_logits(logit, y, reduction="none")).sum() / x.shape[0]
        loss_adv = torch.zeros((), dtype=self.dtype, device=self.device)
        if c["lambda_max"] > 0:
            la = self.net.adversary(grad_reverse(z, lam), y)
            v = self.group_weights[a]
            loss_adv = (v * F.cross_entropy(la, a, reduction="none")).sum() / v.sum()
        return loss_y, kl, loss_adv

    def fit(self, X, y, a, X_val=None, y_val=None, a_val=None, verbose=True, epoch_callback=None):
        c = self.config
        X = torch.as_tensor(np.asarray(X), dtype=self.dtype, device=self.device)
        y = torch.as_tensor(np.asarray(y), dtype=self.dtype, device=self.device)
        a = torch.as_tensor(np.asarray(a), dtype=torch.long, device=self.device)
        n = len(X)
        counts = torch.bincount(a, minlength=c["n_groups"]).to(self.dtype)
        if c["balanced_adversary"]:
            self.group_weights = torch.where(counts > 0, n / (c["n_groups"] * counts.clamp(min=1)), torch.zeros_like(counts))
        main = [p for nme, p in self.net.named_parameters() if not nme.startswith("adv")]
        adv = [p for nme, p in self.net.named_parameters() if nme.startswith("adv")]
        opt = torch.optim.AdamW([{"params": main, "lr": c["lr"], "weight_decay": c["weight_decay"]},
                                 {"params": adv, "lr": c["adv_lr"], "weight_decay": 0.0}])
        steps = math.ceil(n / c["batch_size"]) * c["epochs"]
        step, hist = 0, {k: [] for k in ("epoch", "lambda", "loss_y", "kl", "loss_adv", "val_auc")}
        gen = torch.Generator(device="cpu").manual_seed(c["seed"])
        for ep in range(c["epochs"]):
            self.net.train()
            perm = torch.randperm(n, generator=gen).to(self.device)
            tot = [0.0, 0.0, 0.0]
            nb = 0
            lam = 0.0
            for i in range(0, n, c["batch_size"]):
                idx = perm[i:i + c["batch_size"]]
                lam = self.lambda_at(step / max(1, steps))
                ly, kl, la = self.losses(X[idx], y[idx], a[idx], lam)
                loss = ly + c["beta"] * kl + la      # GRL turns +la into -lam*la for the encoder
                opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), c["max_grad_norm"])
                opt.step()
                # extra adversary-only updates on the same (detached) representation, as in the NumPy model
                for _ in range(c["adv_steps"] - 1 if c["lambda_max"] > 0 else 0):
                    la_ = self.net.adversary(self._last_z, y[idx])
                    v_ = self.group_weights[a[idx]]
                    l_adv = (v_ * F.cross_entropy(la_, a[idx], reduction="none")).sum() / v_.sum()
                    opt.zero_grad(set_to_none=True)
                    l_adv.backward()
                    nn.utils.clip_grad_norm_(adv, c["max_grad_norm"])
                    opt.step()
                tot = [tot[0] + float(ly), tot[1] + float(kl), tot[2] + float(la)]
                nb += 1
                step += 1
            hist["epoch"].append(ep + 1)
            hist["lambda"].append(lam)
            for k, v in zip(("loss_y", "kl", "loss_adv"), tot):
                hist[k].append(v / nb)
            auc = float("nan")
            if X_val is not None and y_val is not None:
                from sklearn.metrics import roc_auc_score
                auc = float(roc_auc_score(np.asarray(y_val), self.predict_proba(X_val)))
            hist["val_auc"].append(auc)
            if epoch_callback is not None:
                for k, v in (epoch_callback(self, ep + 1) or {}).items():
                    hist.setdefault(k, []).append(v)
            if verbose:
                print(f"      [torch] epoch {ep + 1} lambda={lam:.3f} loss_y={hist['loss_y'][-1]:.4f} val_auc={auc:.4f}")
        hist["selected_epoch"] = c["epochs"]
        self.history = hist
        return hist

    @torch.no_grad()
    def predict_proba(self, X):
        self.net.eval()
        X = torch.as_tensor(np.asarray(X), dtype=self.dtype, device=self.device)
        _, mu = self.net.encode(X)
        return torch.sigmoid(self.net.head(mu)).cpu().numpy().astype(np.float64)

    @torch.no_grad()
    def encode(self, X):
        self.net.eval()
        X = torch.as_tensor(np.asarray(X), dtype=self.dtype, device=self.device)
        return self.net.encode(X)[1].cpu().numpy()

    def n_params(self, inference_only=True):
        names = ["enc1", "enc2", "mu", "head1", "head2"] + ([] if inference_only else ["lv", "adv1", "adv2"])
        return int(sum(p.numel() for nme in names for p in getattr(self.net, nme).parameters()))

    def inference_size_kb(self):
        return self.n_params() * (8 if self.dtype == torch.float64 else 4) / 1024.0

    def save(self, path):
        torch.save({"config": self.config, "state_dict": self.net.state_dict()}, path)
