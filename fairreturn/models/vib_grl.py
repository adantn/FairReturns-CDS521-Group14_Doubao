"""Variational Information Bottleneck + Gradient Reversal, implemented in NumPy (no deep-learning framework).

    x -> encoder MLP -> (mu, log sigma^2) -> z = mu + sigma * eps -> return head -> P(return)
                                              z --GRL(-lambda)--> country adversary -> P(country group)

    encoder and head minimise   BCE(y, y_hat) + beta * KL(q(z|x) || N(0, I)) - lambda * CE(a, a_hat)
    the adversary minimises     CE(a, a_hat)

The gradient reversal layer is the identity in the forward pass and multiplies the gradient by -lambda
in the backward pass, so one simultaneous update trains the adversary to detect the country group while
the encoder learns to hide it. The adversary gets `adv_steps` updates per mini-batch (default 3): an
adversary that lags behind is easily fooled, the encoder then chases it, training oscillates and a fresh
probe still finds the country. With condition_on_y=True the adversary also sees the true label, which
targets Equalized Odds (z independent of country given the outcome); with False it targets demographic
parity. lambda ramps from 0 to lambda_max with the schedule of Ganin & Lempitsky (2015). The adversary
uses class-balanced cross-entropy so that small country groups are not ignored.

The model is meant to be trained on inputs without the fields that the destination sets (country, postage):
if such a proxy is left in, the adversary pushes the encoder to use it *in reverse* to equalise error
rates, so group gaps shrink but the same order still gets a different score in another country.

At inference only the encoder mean and the return head are used; the logvar head and the adversary
are discarded, so debiasing adds no inference cost.

Ablations are special cases: beta=0, lambda_max=0 -> plain MLP; beta>0 -> VIB only; lambda_max>0 -> GRL only.
References: Alemi et al. (2017) VIB; Ganin & Lempitsky (2015) GRL; Zhang, Lemoine & Mitchell (2018).
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np

MAIN_PARAMS = ["enc1_W", "enc1_b", "enc2_W", "enc2_b", "mu_W", "mu_b", "lv_W", "lv_b",
               "head1_W", "head1_b", "head2_W", "head2_b"]
ADV_PARAMS = ["adv1_W", "adv1_b", "adv2_W", "adv2_b"]
INFERENCE_PARAMS = ["enc1_W", "enc1_b", "enc2_W", "enc2_b", "mu_W", "mu_b",
                    "head1_W", "head1_b", "head2_W", "head2_b"]
LOGVAR_MIN, LOGVAR_MAX = -8.0, 8.0


def _relu(x):
    return np.maximum(x, 0)


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -40, 40)))


class _Adam:
    """Adam with optional decoupled weight decay on weight matrices (AdamW)."""

    def __init__(self, params: dict, keys: list[str], lr: float, b1=0.9, b2=0.999, eps=1e-8, weight_decay=0.0):
        self.keys, self.lr, self.b1, self.b2, self.eps = keys, lr, b1, b2, eps
        self.weight_decay = weight_decay
        self.m = {k: np.zeros_like(params[k]) for k in keys}
        self.v = {k: np.zeros_like(params[k]) for k in keys}
        self.t = 0

    def step(self, params: dict, grads: dict):
        self.t += 1
        c1 = 1 - self.b1 ** self.t
        c2 = 1 - self.b2 ** self.t
        for k in self.keys:
            g = grads.get(k)
            if g is None:
                continue
            self.m[k] = self.b1 * self.m[k] + (1 - self.b1) * g
            self.v[k] = self.b2 * self.v[k] + (1 - self.b2) * (g * g)
            params[k] -= self.lr * (self.m[k] / c1) / (np.sqrt(self.v[k] / c2) + self.eps)
            if self.weight_decay and k.endswith("_W"):
                params[k] -= self.lr * self.weight_decay * params[k]


def _clip(grads: dict, max_norm: float):
    total = math.sqrt(sum(float(np.sum(g * g)) for g in grads.values()))
    if total > max_norm:
        s = max_norm / (total + 1e-12)
        for k in grads:
            grads[k] *= s
    return total


class FairReturnModel:
    """Same public interface as the scikit-learn baselines: fit / predict_proba, plus encode / save / load."""

    def __init__(self, input_dim: int, n_groups: int, z_dim: int = 16, hidden=(64, 32), head_hidden: int = 16,
                 adv_hidden: int = 32, beta: float = 1e-3, lambda_max: float = 1.0, condition_on_y: bool = True,
                 lr: float = 1e-3, adv_lr: float | None = None, adv_steps: int = 3, batch_size: int = 1024,
                 epochs: int = 15, pos_weight: float = 1.0, balanced_adversary: bool = True,
                 max_grad_norm: float = 5.0, warmup: float = 0.1, weight_decay: float = 0.0,
                 select_epoch: str = "last", seed: int = 0, dtype: str = "float32"):
        self.config = dict(input_dim=int(input_dim), n_groups=int(n_groups), z_dim=int(z_dim),
                           hidden=[int(h) for h in hidden], head_hidden=int(head_hidden), adv_hidden=int(adv_hidden),
                           beta=float(beta), lambda_max=float(lambda_max), condition_on_y=bool(condition_on_y),
                           lr=float(lr), adv_lr=float(adv_lr if adv_lr is not None else lr), adv_steps=int(adv_steps),
                           batch_size=int(batch_size), epochs=int(epochs), pos_weight=float(pos_weight),
                           balanced_adversary=bool(balanced_adversary), max_grad_norm=float(max_grad_norm),
                           warmup=float(warmup), weight_decay=float(weight_decay), select_epoch=select_epoch,
                           seed=int(seed), dtype=dtype)
        self.dtype = np.dtype(dtype)
        self.rng = np.random.default_rng(seed)
        self.params = self._init_params()
        self.group_weights = np.ones(n_groups, dtype=self.dtype)
        self.history: dict = {}

    # ------------------------------------------------------------------ parameters
    def _init_params(self) -> dict:
        c, r, dt = self.config, self.rng, self.dtype
        h1, h2 = c["hidden"]
        k, hh, ah, g = c["z_dim"], c["head_hidden"], c["adv_hidden"], c["n_groups"]
        adv_in = k + (1 if c["condition_on_y"] else 0)

        def he(n_in, n_out):
            return (r.standard_normal((n_in, n_out)) * math.sqrt(2.0 / n_in)).astype(dt)

        def lin(n_in, n_out):
            return (r.standard_normal((n_in, n_out)) * math.sqrt(1.0 / n_in)).astype(dt)

        z = lambda n: np.zeros(n, dtype=dt)  # noqa: E731
        return {
            "enc1_W": he(c["input_dim"], h1), "enc1_b": z(h1), "enc2_W": he(h1, h2), "enc2_b": z(h2),
            "mu_W": lin(h2, k), "mu_b": z(k), "lv_W": (lin(h2, k) * 0.1).astype(dt), "lv_b": np.full(k, -2.0, dt),
            "head1_W": he(k, hh), "head1_b": z(hh), "head2_W": lin(hh, 1), "head2_b": z(1),
            "adv1_W": he(adv_in, ah), "adv1_b": z(ah), "adv2_W": lin(ah, g), "adv2_b": z(g),
        }

    # ------------------------------------------------------------------ core forward / backward
    def _encode(self, x):
        P = self.params
        a1 = x @ P["enc1_W"] + P["enc1_b"]
        h1 = _relu(a1)
        a2 = h1 @ P["enc2_W"] + P["enc2_b"]
        h2 = _relu(a2)
        mu = h2 @ P["mu_W"] + P["mu_b"]
        return a1, h1, a2, h2, mu

    def _head(self, z):
        P = self.params
        c1a = z @ P["head1_W"] + P["head1_b"]
        c1 = _relu(c1a)
        logit = (c1 @ P["head2_W"] + P["head2_b"])[:, 0]
        return c1a, c1, logit

    def _adversary(self, u):
        P = self.params
        d1a = u @ P["adv1_W"] + P["adv1_b"]
        d1 = _relu(d1a)
        la = d1 @ P["adv2_W"] + P["adv2_b"]
        return d1a, d1, la

    def _adv_input(self, z, y):
        return np.concatenate([z, y[:, None].astype(z.dtype)], axis=1) if self.config["condition_on_y"] else z

    def forward_backward(self, x, y, a, lam: float, eps=None, with_adversary: bool = True):
        """One mini-batch. Returns (losses, grads_main, grads_adv).

        grads_main = d/dtheta [L_y + beta*KL - lam*L_adv]   (encoder and head; GRL flips the adversary term)
        grads_adv  = d/dphi   [L_adv]                        (adversary)
        """
        c, P = self.config, self.params
        B = x.shape[0]
        beta = c["beta"]
        a1, h1, a2, h2, mu = self._encode(x)
        if beta > 0:
            lv_raw = h2 @ P["lv_W"] + P["lv_b"]
            lv = np.clip(lv_raw, LOGVAR_MIN, LOGVAR_MAX)
            std = np.exp(0.5 * lv)
            if eps is None:
                eps = self.rng.standard_normal(mu.shape).astype(self.dtype)
            z = mu + std * eps
            kl = float(np.sum(0.5 * (mu * mu + np.exp(lv) - 1.0 - lv)) / B)
        else:
            z, kl = mu, 0.0
        c1a, c1, logit = self._head(z)
        p = _sigmoid(logit)
        w = np.where(y > 0.5, c["pos_weight"], 1.0).astype(self.dtype)
        bce = np.maximum(logit, 0) - logit * y + np.log1p(np.exp(-np.abs(logit)))
        loss_y = float(np.sum(w * bce) / B)

        g = {}
        dlogit = (w * (p - y) / B).astype(self.dtype)
        g["head2_W"] = c1.T @ dlogit[:, None]
        g["head2_b"] = np.array([dlogit.sum()], dtype=self.dtype)
        dc1a = (dlogit[:, None] @ P["head2_W"].T) * (c1a > 0)
        g["head1_W"] = z.T @ dc1a
        g["head1_b"] = dc1a.sum(0)
        dz = dc1a @ P["head1_W"].T

        loss_adv, ga = 0.0, {}
        if with_adversary and c["lambda_max"] > 0:
            u = self._adv_input(z, y)
            d1a, d1, la = self._adversary(u)
            la_shift = la - la.max(axis=1, keepdims=True)
            ex = np.exp(la_shift)
            sm = ex / ex.sum(axis=1, keepdims=True)
            v = self.group_weights[a]
            vsum = float(v.sum()) or 1.0
            ce = -np.log(sm[np.arange(B), a] + 1e-12)
            loss_adv = float(np.sum(v * ce) / vsum)
            onehot = np.zeros_like(sm)
            onehot[np.arange(B), a] = 1.0
            dla = (v[:, None] * (sm - onehot) / vsum).astype(self.dtype)
            ga["adv2_W"] = d1.T @ dla
            ga["adv2_b"] = dla.sum(0)
            dd1a = (dla @ P["adv2_W"].T) * (d1a > 0)
            ga["adv1_W"] = u.T @ dd1a
            ga["adv1_b"] = dd1a.sum(0)
            du = dd1a @ P["adv1_W"].T
            dz = dz - lam * du[:, : c["z_dim"]]          # gradient reversal

        if beta > 0:
            dmu = dz + beta * mu / B
            dlv = dz * eps * 0.5 * std + beta * 0.5 * (np.exp(lv) - 1.0) / B
            dlv = dlv * ((lv_raw > LOGVAR_MIN) & (lv_raw < LOGVAR_MAX))
            g["lv_W"] = h2.T @ dlv
            g["lv_b"] = dlv.sum(0)
            dh2 = dmu @ P["mu_W"].T + dlv @ P["lv_W"].T
        else:
            dmu = dz
            dh2 = dmu @ P["mu_W"].T
        g["mu_W"] = h2.T @ dmu
        g["mu_b"] = dmu.sum(0)
        da2 = dh2 * (a2 > 0)
        g["enc2_W"] = h1.T @ da2
        g["enc2_b"] = da2.sum(0)
        da1 = (da2 @ P["enc2_W"].T) * (a1 > 0)
        g["enc1_W"] = x.T @ da1
        g["enc1_b"] = da1.sum(0)
        losses = {"loss_y": loss_y, "kl": kl, "loss_adv": loss_adv,
                  "objective": loss_y + beta * kl - lam * loss_adv}
        return losses, g, ga, (z, y, a)

    def _adv_only_step(self, z, y, a, opt):
        """Extra adversary update on a fixed representation (adv_steps > 1)."""
        P = self.params
        B = z.shape[0]
        u = self._adv_input(z, y)
        d1a, d1, la = self._adversary(u)
        ex = np.exp(la - la.max(axis=1, keepdims=True))
        sm = ex / ex.sum(axis=1, keepdims=True)
        v = self.group_weights[a]
        vsum = float(v.sum()) or 1.0
        onehot = np.zeros_like(sm)
        onehot[np.arange(B), a] = 1.0
        dla = (v[:, None] * (sm - onehot) / vsum).astype(self.dtype)
        ga = {"adv2_W": d1.T @ dla, "adv2_b": dla.sum(0)}
        dd1a = (dla @ P["adv2_W"].T) * (d1a > 0)
        ga["adv1_W"] = u.T @ dd1a
        ga["adv1_b"] = dd1a.sum(0)
        _clip(ga, self.config["max_grad_norm"])
        opt.step(P, ga)

    # ------------------------------------------------------------------ training
    def lambda_at(self, progress: float) -> float:
        """0 during warm-up (the predictor and adversary learn first), then the Ganin & Lempitsky ramp."""
        w = self.config["warmup"]
        if progress < w:
            return 0.0
        q = (progress - w) / max(1e-9, 1.0 - w)
        return self.config["lambda_max"] * (2.0 / (1.0 + math.exp(-10.0 * q)) - 1.0)

    def fit(self, X, y, a, X_val=None, y_val=None, a_val=None, verbose: bool = True, epoch_callback=None):
        """epoch_callback(model, epoch) -> dict of extra numbers to record in the history after every epoch."""
        c = self.config
        X = np.ascontiguousarray(X, dtype=self.dtype)
        y = np.asarray(y, dtype=self.dtype)
        a = np.asarray(a, dtype=np.int64)
        n = len(X)
        counts = np.bincount(a, minlength=c["n_groups"]).astype(float)
        if c["balanced_adversary"]:
            wts = np.where(counts > 0, n / (c["n_groups"] * np.maximum(counts, 1)), 0.0)
        else:
            wts = np.ones(c["n_groups"])
        self.group_weights = wts.astype(self.dtype)
        opt_main = _Adam(self.params, MAIN_PARAMS, c["lr"], weight_decay=c["weight_decay"])
        opt_adv = _Adam(self.params, ADV_PARAMS, c["adv_lr"])
        bs = c["batch_size"]
        steps_per_epoch = math.ceil(n / bs)
        total = c["epochs"] * steps_per_epoch
        step = 0
        hist = {k: [] for k in ("epoch", "lambda", "loss_y", "kl", "loss_adv", "val_auc", "seconds")}
        best = (-np.inf, None, None)   # (val_auc, epoch, params snapshot) for select_epoch='best_val'
        for ep in range(c["epochs"]):
            t0 = time.time()
            perm = self.rng.permutation(n)
            sums = {"loss_y": 0.0, "kl": 0.0, "loss_adv": 0.0}
            lam = 0.0
            for i in range(0, n, bs):
                idx = perm[i:i + bs]
                lam = self.lambda_at(step / max(1, total))
                losses, g, ga, (z, yb, ab) = self.forward_backward(X[idx], y[idx], a[idx], lam)
                _clip(g, c["max_grad_norm"])
                opt_main.step(self.params, g)
                if ga:
                    _clip(ga, c["max_grad_norm"])
                    opt_adv.step(self.params, ga)
                    for _ in range(c["adv_steps"] - 1):
                        self._adv_only_step(z, yb, ab, opt_adv)
                for k in sums:
                    sums[k] += losses[k]
                step += 1
            hist["epoch"].append(ep + 1)
            hist["lambda"].append(round(lam, 4))
            for k in sums:
                hist[k].append(sums[k] / steps_per_epoch)
            val_auc = float("nan")
            if X_val is not None and y_val is not None and len(np.unique(y_val)) > 1:
                from sklearn.metrics import roc_auc_score
                val_auc = float(roc_auc_score(np.asarray(y_val), self.predict_proba(X_val)))
            hist["val_auc"].append(val_auc)
            hist["seconds"].append(round(time.time() - t0, 2))
            if epoch_callback is not None:
                for k, v in (epoch_callback(self, ep + 1) or {}).items():
                    hist.setdefault(k, []).append(v)
            # only epochs where the fairness pressure is (almost) fully on may be selected
            eligible = c["lambda_max"] == 0 or lam >= 0.9 * c["lambda_max"]
            if c["select_epoch"] == "best_val" and eligible and np.isfinite(val_auc) and val_auc > best[0]:
                best = (val_auc, ep + 1, {k: v.copy() for k, v in self.params.items()})
            if verbose and (ep == 0 or (ep + 1) % 5 == 0 or ep + 1 == c["epochs"]):
                print(f"      epoch {ep + 1:3d}  lambda={lam:.3f}  loss_y={hist['loss_y'][-1]:.4f}  "
                      f"kl={hist['kl'][-1]:.2f}  loss_adv={hist['loss_adv'][-1]:.4f}  "
                      f"val_auc={val_auc:.4f}  ({hist['seconds'][-1]:.1f}s)")
        if c["select_epoch"] == "best_val" and best[2] is not None:
            self.params = best[2]
            hist["selected_epoch"] = best[1]
        else:
            hist["selected_epoch"] = c["epochs"]
        self.history = hist
        return hist

    # ------------------------------------------------------------------ inference
    def predict_logit(self, X, batch_size: int = 16384):
        X = np.asarray(X, dtype=self.dtype)
        out = np.empty(len(X), dtype=np.float64)
        for i in range(0, len(X), batch_size):
            mu = self._encode(X[i:i + batch_size])[-1]
            out[i:i + batch_size] = self._head(mu)[-1]
        return out

    def predict_proba(self, X):
        return _sigmoid(self.predict_logit(X))

    def encode(self, X, batch_size: int = 16384):
        X = np.asarray(X, dtype=self.dtype)
        return np.concatenate([self._encode(X[i:i + batch_size])[-1] for i in range(0, len(X), batch_size)]) \
            if len(X) else np.zeros((0, self.config["z_dim"]), dtype=self.dtype)

    def adversary_proba(self, X, y):
        """Country-group probabilities from the training-time adversary (diagnostics only)."""
        z = self.encode(X)
        la = self._adversary(self._adv_input(z, np.asarray(y, dtype=self.dtype)))[-1]
        ex = np.exp(la - la.max(axis=1, keepdims=True))
        return ex / ex.sum(axis=1, keepdims=True)

    def n_params(self, inference_only: bool = True) -> int:
        keys = INFERENCE_PARAMS if inference_only else MAIN_PARAMS + ADV_PARAMS
        return int(sum(self.params[k].size for k in keys))

    # ------------------------------------------------------------------ persistence
    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        arrays = {k: v for k, v in self.params.items()}
        arrays["__group_weights__"] = self.group_weights
        arrays["__config__"] = np.array(json.dumps(self.config))
        arrays["__history__"] = np.array(json.dumps(self.history, default=float))
        np.savez_compressed(path, **arrays)

    @classmethod
    def load(cls, path) -> "FairReturnModel":
        with np.load(path, allow_pickle=False) as f:
            config = json.loads(str(f["__config__"]))
            obj = cls(**config)
            for k in obj.params:
                obj.params[k] = f[k].astype(obj.dtype)
            obj.group_weights = f["__group_weights__"].astype(obj.dtype)
            obj.history = json.loads(str(f["__history__"]))
        return obj

    def inference_size_kb(self) -> float:
        return self.n_params(inference_only=True) * self.dtype.itemsize / 1024.0
