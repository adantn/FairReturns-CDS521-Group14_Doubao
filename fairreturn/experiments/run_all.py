"""Train and evaluate every method on the real Online Retail II data.

    python -m fairreturn.experiments.run_all                     # main results: all methods, 3 seeds (~10 min)
    python -m fairreturn.experiments.run_all --quick             # 3-minute smoke test -> outputs/results/quick_main/
    python -m fairreturn.experiments.run_all --sweep             # beta x lambda trade-off for the Pareto figure
    python -m fairreturn.experiments.run_all --inject-bias 0.03 --bias-groups Germany France
                                                                 # stress test: synthetic historical label bias
    python -m fairreturn.experiments.run_all --methods gbdt_aware vib_grl --seeds 1
    python -m fairreturn.experiments.run_all --product-tags llm --methods gbdt_blind vib_grl --bootstrap 0
                                                                 # the same models with the LLM product tags

Evaluation is on the time-based test split (Jun-Oct 2011) with the real labels. Main outputs:
    outputs/results/summary.csv|.md|.json   mean +- std over seeds (+ customer-bootstrap 95% CIs, seed 0)
    outputs/results/group_metrics.csv       per country group: TPR, FPR, flag rate, calibration (seed 0)
    outputs/results/seeds.csv               every run
    outputs/results/input_probe.json        how much country information the input features carry
    outputs/results/preds/<method>.npz      seed-0 scores on val / test / counterfactual twins
    outputs/models/<method>/                seed-0 trained models used by the web app
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from fairreturn import config as cfg
from fairreturn.data.features import FeatureBuilder
from fairreturn.evaluation import metrics as M
from fairreturn.models import registry

NEURAL_DEFAULTS = {"epochs": 12, "select_epoch": "best_val", "weight_decay": 1e-4, "adv_steps": 3}
GROUP_INDEX = {g: i for i, g in enumerate(cfg.GROUPS)}


# ----------------------------------------------------------------------------- data
def apply_llm_tags(frame: pd.DataFrame, allow_mock: bool = False) -> tuple[pd.DataFrame, dict]:
    """Replace the keyword-rule product tags by the LLM tags (outputs/genai/product_tags_llm.csv) wherever the
    LLM answer was valid. Tags are product-level, so this needs no rebuild of data/processed."""
    if not cfg.LLM_TAGS_FILE.exists():
        raise SystemExit("no LLM tags yet - run `python -m fairreturn.genai.product_tagger --backend api` first")
    t = pd.read_csv(cfg.LLM_TAGS_FILE, dtype={"stock_code": str}, keep_default_na=False)
    if t["tag_source"].astype(str).str.endswith("mock").any() and not allow_mock:
        raise SystemExit("product_tags_llm.csv comes from the MOCK backend - rerun the tagger with --backend api "
                         "(or pass --allow-mock-tags for a pipeline test)")
    t = t[t["llm_valid"].astype(int) == 1].set_index("stock_code")
    hit = frame["stock_code"].isin(t.index)
    out = frame.copy()
    for col in ("tag_category", "tag_material"):
        out.loc[hit, col] = out.loc[hit, "stock_code"].map(t[col]).astype(str)
    for col in ("tag_fragile", "tag_seasonal"):
        out.loc[hit, col] = out.loc[hit, "stock_code"].map(t[col]).astype(float)
    return out, {"lines_with_llm_tags": float(hit.mean()), "products_with_llm_tags": int(len(t))}


def load_data(max_train: int | None = None, seed: int = cfg.SEED, product_tags: str = "rules",
              allow_mock_tags: bool = False):
    for f in (cfg.LINES_FILE, cfg.TWINS_FILE, cfg.META_FILE):
        if not f.exists():
            raise SystemExit(f"{f} missing - run `python -m fairreturn.data.prepare` first")
    lines = pd.read_pickle(cfg.LINES_FILE)
    twins = pd.read_pickle(cfg.TWINS_FILE)
    if product_tags == "llm":
        lines, info = apply_llm_tags(lines, allow_mock_tags)
        twins, _ = apply_llm_tags(twins, allow_mock_tags)
        print(f"  LLM product tags: {info['products_with_llm_tags']} products, "
              f"{info['lines_with_llm_tags']:.1%} of lines (keyword rules for the rest)")
    meta = json.loads(cfg.META_FILE.read_text(encoding="utf-8"))
    D = {s: lines[lines["split"] == s].reset_index(drop=True) for s in ("train", "val", "test")}
    if max_train and len(D["train"]) > max_train:
        D["train"] = D["train"].sample(max_train, random_state=seed).reset_index(drop=True)
    D["twin"] = twins.set_index("twin_of").loc[D["test"]["line_id"]].reset_index()
    return D, meta


def codes(frame: pd.DataFrame) -> np.ndarray:
    return frame["group"].map(GROUP_INDEX).to_numpy()


def inject_label_bias(train: pd.DataFrame, rate: float, groups: list[str], seed: int) -> np.ndarray:
    """Historical label bias: a share `rate` of NON-returned lines from `groups` is wrongly labelled
    as returned (e.g. 'suspicious return' flags). Only training labels change; evaluation stays clean."""
    rng = np.random.default_rng(seed + 991)
    y = train["returned"].to_numpy().astype(float).copy()
    m = train["group"].isin(groups).to_numpy() & (y == 0)
    flip = m & (rng.random(len(y)) < rate)
    y[flip] = 1.0
    print(f"  injected label bias: {int(flip.sum()):,} training lines from {groups} relabelled as returned")
    return y


# ----------------------------------------------------------------------------- evaluation
def customer_bootstrap(test: pd.DataFrame, p, p_twin, thr, B: int, seed: int) -> dict:
    """95% CIs for AUC, EO gap and flip rate, resampling customers (lines of one customer are correlated)."""
    rng = np.random.default_rng(seed + 7)
    cust = test["customer_id"].to_numpy()
    uniq, inv = np.unique(cust, return_inverse=True)
    members = [np.flatnonzero(inv == i) for i in range(len(uniq))]
    y, g, cu = test["returned"].to_numpy(), test["group"].to_numpy(), test["customer_id"].to_numpy()
    stats = {"auc": [], "eo_gap": [], "fpr_gap": [], "tpr_gap": [], "cf_flip_rate": []}
    for _ in range(B):
        idx = np.concatenate([members[i] for i in rng.integers(0, len(uniq), len(uniq))])
        acc = M.accuracy_metrics(y[idx], p[idx])
        gaps = M.fairness_gaps(M.group_table(y[idx], p[idx], g[idx], thr, cu[idx]))
        stats["auc"].append(acc["auc"])
        for k in ("eo_gap", "fpr_gap", "tpr_gap"):
            stats[k].append(gaps[k])
        stats["cf_flip_rate"].append(M.counterfactual_metrics(p[idx], p_twin[idx], thr)["cf_flip_rate"])
    return {f"{k}_ci": [float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))] for k, v in stats.items()}


def latency_ms_per_1k(predict, X) -> float:
    Xs = X[:1000]
    predict(Xs)
    t0 = time.perf_counter()
    for _ in range(5):
        predict(Xs)
    return (time.perf_counter() - t0) / 5 * 1000.0


def evaluate(model, X: dict, D: dict, is_neural: bool, seed: int, bootstrap: int = 0) -> tuple[dict, pd.DataFrame, dict]:
    p = {s: model.predict_proba(X[s]) for s in ("val", "test", "twin")}
    thr = M.threshold_top_share(p["val"])
    test = D["test"]
    res = {"threshold": thr}
    res.update(M.accuracy_metrics(test["returned"], p["test"]))
    gt = M.group_table(test["returned"], p["test"], test["group"], thr, test["customer_id"])
    gaps = M.fairness_gaps(gt)
    res.update({k: v for k, v in gaps.items() if not isinstance(v, list)})
    res.update(M.counterfactual_metrics(p["test"], p["twin"], thr))
    res.update(M.probe_output(p["val"], D["val"]["returned"].to_numpy(), codes(D["val"]),
                              p["test"], test["returned"].to_numpy(), codes(test), seed=seed))
    if is_neural:
        res.update(M.probe_representation(model.encode(X["train"]), codes(D["train"]),
                                           model.encode(X["test"]), codes(test), seed=seed))
        res["params"] = model.n_params(inference_only=True)
        res["size_kb"] = model.inference_size_kb()
        res["selected_epoch"] = model.history.get("selected_epoch")
    else:
        res["params"] = model.n_params()
        res["size_kb"] = model.size_kb()
    res["latency_ms_1k"] = latency_ms_per_1k(model.predict_proba, X["test"])
    if bootstrap:
        res.update(customer_bootstrap(test, p["test"], p["twin"], thr, bootstrap, seed))
    gt["threshold"] = thr
    return res, gt, p


# ----------------------------------------------------------------------------- one method
def _val_gaps_callback(X_val, val: pd.DataFrame):
    """Validation fairness after every epoch, recorded in the training history (for the training figure).
    Only validation data is used, so it can never leak test information into training or selection."""
    y, g, c = val["returned"].to_numpy(), val["group"].to_numpy(), val["customer_id"].to_numpy()

    def cb(model, epoch):
        p = model.predict_proba(X_val)
        gaps = M.fairness_gaps(M.group_table(y, p, g, M.threshold_top_share(p), c))
        return {"val_fpr_gap": gaps["fpr_gap"], "val_tpr_gap": gaps["tpr_gap"]}
    return cb


def run_method(method, D, X_cache, y_train, seed, epochs, overrides, bootstrap, save_dir, tag, verbose,
               preds_dir: Path | None = None):
    fs = method.feature_set
    if fs not in X_cache:
        fb = FeatureBuilder(fs).fit(D["train"])
        X_cache[fs] = (fb, {s: fb.transform(D[s]) for s in ("train", "val", "test", "twin")})
    fb, X = X_cache[fs]
    is_neural = method.family != "sklearn"
    ov = dict(NEURAL_DEFAULTS, **(overrides or {})) if is_neural else dict(overrides or {})
    if is_neural and epochs:
        ov["epochs"] = epochs
    model = registry.build_model(method, fb.n_features, len(cfg.GROUPS), seed, overrides=ov)
    t0 = time.time()
    if is_neural:
        model.fit(X["train"], y_train, codes(D["train"]), X["val"], D["val"]["returned"].to_numpy(), verbose=verbose,
                  epoch_callback=_val_gaps_callback(X["val"], D["val"]))
    else:
        model.fit(X["train"], y_train, codes(D["train"]), X["val"], D["val"]["returned"].to_numpy(), verbose=verbose)
    train_s = time.time() - t0
    res, gt, preds = evaluate(model, X, D, is_neural, seed, bootstrap)
    res["train_seconds"] = train_s
    if save_dir is not None:
        info = {"name": method.name, "label": method.label, "family": method.family, "role": method.role,
                "feature_set": fs, "params": {**method.params, **ov}, "seed": seed, "tag": tag,
                "metrics": {k: v for k, v in res.items() if isinstance(v, (int, float)) and v is not None},
                "history": getattr(model, "history", {}), "trained": time.strftime("%Y-%m-%d %H:%M:%S")}
        registry.save_artifacts(method.name, model, fb, X["train"].mean(axis=0), info, save_dir)
        cfg.PREDS_DIR.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cfg.PREDS_DIR / f"{method.name}.npz", val=preds["val"].astype(np.float32),
                            test=preds["test"].astype(np.float32), twin=preds["twin"].astype(np.float32))
    elif preds_dir is not None:
        preds_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(preds_dir / f"{method.name}.npz", val=preds["val"].astype(np.float32),
                            test=preds["test"].astype(np.float32), twin=preds["twin"].astype(np.float32))
    return res, gt


# ----------------------------------------------------------------------------- tables
SUMMARY_COLS = ["auc", "pr_auc", "brier", "eo_gap", "tpr_gap", "fpr_gap", "dp_gap", "calibration_gap_max",
                "cf_flip_rate", "cf_mean_abs_delta", "output_probe", "probe_best", "chance", "params", "size_kb",
                "latency_ms_1k", "train_seconds"]


def fmt(m, s, nd=3):
    if m is None or (isinstance(m, float) and np.isnan(m)):
        return "—"
    return f"{m:.{nd}f}" if (s is None or np.isnan(s) or s < 10 ** -(nd + 1)) else f"{m:.{nd}f} ± {s:.{nd}f}"


def write_summary(raw: pd.DataFrame, out_dir: Path, extra_ci: dict) -> pd.DataFrame:
    cols = [c for c in SUMMARY_COLS if c in raw.columns]
    g = raw.groupby(["method", "label", "role"], sort=False)[cols]
    mean, std = g.mean(), g.std()
    summ = mean.add_suffix("_mean").join(std.add_suffix("_std")).reset_index()
    summ["n_seeds"] = raw.groupby(["method"], sort=False).size().to_numpy()
    for m, ci in extra_ci.items():
        for k, v in ci.items():
            summ.loc[summ["method"] == m, k] = json.dumps([round(x, 4) for x in v])
    summ.to_csv(out_dir / "summary.csv", index=False)
    (out_dir / "summary.json").write_text(summ.to_json(orient="records", indent=1), encoding="utf-8")

    lines = ["| Method | AUC ↑ | PR-AUC ↑ | ΔFPR ↓ | ΔTPR ↓ | ΔDP ↓ | CF flip ↓ | Output probe ↓ | z probe ↓ "
             "| Size (KB) | ms / 1k |", "|" + "---|" * 11]
    for _, r in summ.iterrows():
        lines.append("| " + " | ".join([
            r["label"], fmt(r["auc_mean"], r["auc_std"]), fmt(r["pr_auc_mean"], r["pr_auc_std"]),
            fmt(r["fpr_gap_mean"], r["fpr_gap_std"]), fmt(r["tpr_gap_mean"], r["tpr_gap_std"]),
            fmt(r["dp_gap_mean"], r["dp_gap_std"]), fmt(r["cf_flip_rate_mean"], r["cf_flip_rate_std"], 4),
            fmt(r["output_probe_mean"], r["output_probe_std"]),
            fmt(r.get("probe_best_mean"), r.get("probe_best_std")),
            f"{r['size_kb_mean']:.1f}" if pd.notna(r.get("size_kb_mean")) else "—",
            f"{r['latency_ms_1k_mean']:.1f}"]) + " |")
    chance = raw["chance"].dropna().iloc[0] if "chance" in raw and raw["chance"].notna().any() else 1 / len(cfg.GROUPS)
    lines.append(f"\nΔFPR / ΔTPR: largest gap between country groups in false-alarm / hit rate at the high-risk "
                 f"threshold (top {cfg.HIGH_RISK_SHARE:.0%} of validation scores); groups need returns or non-returns "
                 f"from ≥{M.MIN_CUSTOMERS} customers. CF flip: share of test lines whose flag changes when only the "
                 f"country (and postage) changes. Probe: balanced accuracy of recovering the country group "
                 f"(chance = {chance:.3f}).")
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n" + "\n".join(lines))
    return summ


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--methods", nargs="*", default=None, help=f"subset of {list(registry.BY_NAME)}")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=None, help="override epochs for neural models")
    ap.add_argument("--bootstrap", type=int, default=100, help="customer-bootstrap samples for 95%% CIs (seed 0)")
    ap.add_argument("--max-train", type=int, default=None)
    ap.add_argument("--quick", action="store_true", help="smoke test: 60k training lines, 5 epochs, 1 seed; written to outputs/results/quick_*/, models are not saved")
    ap.add_argument("--sweep", action="store_true", help="beta x lambda grid for VIB+GRL (seed 0)")
    ap.add_argument("--inject-bias", type=float, default=0.0)
    ap.add_argument("--bias-groups", nargs="*", default=["Germany", "France"])
    ap.add_argument("--tag", default=None, help="results sub-folder (default: main, sweep or stress_bias_<rate>)")
    ap.add_argument("--no-save", action="store_true", help="do not overwrite outputs/models and preds")
    ap.add_argument("--product-tags", choices=["rules", "llm"], default="rules",
                    help="'llm' swaps in the LLM product tags (results go to outputs/results/llm_tags/)")
    ap.add_argument("--allow-mock-tags", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args()

    if a.quick:
        a.max_train, a.epochs, a.seeds, a.bootstrap = 60000, 5, 1, 0
    tag = a.tag or ("sweep" if a.sweep else (f"stress_bias_{a.inject_bias:g}" if a.inject_bias > 0 else
                                            ("llm_tags" if a.product_tags == "llm" else "main")))
    if a.quick and not a.tag:
        tag = f"quick_{tag}"          # a smoke test never overwrites the real results
    if a.quick:
        print("QUICK MODE: subsampled and few epochs - the numbers only show that the pipeline runs; "
              "adversarial models in particular are far from converged. Do not report them.")
    out_dir = cfg.RESULTS_DIR if tag == "main" else cfg.RESULTS_DIR / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    save_models = tag == "main" and not a.no_save and not a.quick
    D, meta = load_data(a.max_train, product_tags=a.product_tags, allow_mock_tags=a.allow_mock_tags)
    print(f"train {len(D['train']):,} | val {len(D['val']):,} | test {len(D['test']):,} lines "
          f"(+{len(D['twin']):,} counterfactual twins)   tag={tag}")
    X_cache: dict = {}
    t_all = time.time()

    if a.sweep:
        rows = []
        base = registry.get_method("vib_grl")
        for beta in (0.0, 1e-3, 1e-2):
            for lam in (0.0, 0.5, 1.0, 2.0, 4.0):
                print(f"[sweep] beta={beta:g} lambda_max={lam:g}")
                res, _ = run_method(base, D, X_cache, D["train"]["returned"].to_numpy(), 0, a.epochs,
                                    {"beta": beta, "lambda_max": lam}, 0, None, tag, False)
                rows.append({"beta": beta, "lambda_max": lam, **res})
                print(f"   AUC {res['auc']:.4f}  ΔFPR {res['fpr_gap']:.3f}  ΔTPR {res['tpr_gap']:.3f}  "
                      f"flip {res['cf_flip_rate']:.4f}  z-probe {res['probe_best']:.3f}")
        pd.DataFrame(rows).to_csv(out_dir / "sweep.csv", index=False)
        print(f"wrote {out_dir / 'sweep.csv'} in {time.time() - t_all:.0f}s")
        return

    methods = [registry.get_method(m) for m in (a.methods or [m.name for m in registry.METHODS])]
    y_train = D["train"]["returned"].to_numpy().astype(float)
    if a.inject_bias > 0:
        y_train = inject_label_bias(D["train"], a.inject_bias, a.bias_groups, cfg.SEED)

    if tag == "main" and not a.methods:
        inp = {}
        for fs in ("aware", "unaware", "blind"):
            fb = FeatureBuilder(fs).fit(D["train"])
            inp[fs] = M.probe_representation(fb.transform(D["train"]), codes(D["train"]),
                                             fb.transform(D["test"]), codes(D["test"]))
        (out_dir / "input_probe.json").write_text(json.dumps(inp, indent=2), encoding="utf-8")
        print("input-feature probe (country recoverable from the raw inputs):",
              {k: round(v["probe_best"], 3) for k, v in inp.items()})

    raw, group_rows, ci = [], [], {}
    for seed in range(a.seeds):
        for m in methods:
            if m.deterministic and seed > 0:
                prev = next(r for r in raw if r["method"] == m.name and r["seed"] == 0)
                raw.append({**prev, "seed": seed})
                continue
            print(f"[seed {seed}] {m.label}")
            res, gt = run_method(m, D, X_cache, y_train, seed, a.epochs, None,
                                 a.bootstrap if seed == 0 else 0,
                                 cfg.MODELS_DIR if (save_models and seed == 0) else None, tag, not a.quiet,
                                 preds_dir=(out_dir / "preds") if (tag != "main" and seed == 0) else None)
            print(f"   AUC {res['auc']:.4f}  ΔEO {res['eo_gap']:.3f}  ΔDP {res['dp_gap']:.3f}  "
                  f"flip {res['cf_flip_rate']:.4f}  output-probe {res['output_probe']:.3f}"
                  + (f"  z-probe {res['probe_best']:.3f}" if "probe_best" in res else "")
                  + f"  ({res['train_seconds']:.0f}s)")
            raw.append({"method": m.name, "label": m.label, "role": m.role, "seed": seed, **res})
            if seed == 0:
                group_rows.append(gt.assign(method=m.name, label=m.label))
                ci[m.name] = {k: v for k, v in res.items() if k.endswith("_ci")}
    raw_df = pd.DataFrame([{k: v for k, v in r.items() if not k.endswith("_ci")} for r in raw])
    raw_df.to_csv(out_dir / "seeds.csv", index=False)
    pd.concat(group_rows).to_csv(out_dir / "group_metrics.csv", index=False)
    write_summary(raw_df, out_dir, ci)
    run_info = {"tag": tag, "args": vars(a), "neural_defaults": NEURAL_DEFAULTS, "high_risk_share": cfg.HIGH_RISK_SHARE,
                "n_train": len(D["train"]), "n_val": len(D["val"]), "n_test": len(D["test"]),
                "data_meta_created": meta.get("created"), "tags": meta.get("tags"),
                "minutes": round((time.time() - t_all) / 60, 1), "finished": time.strftime("%Y-%m-%d %H:%M:%S")}
    (out_dir / "run_info.json").write_text(json.dumps(run_info, indent=2), encoding="utf-8")
    print(f"\nwrote {out_dir}/summary.md, summary.csv, group_metrics.csv, seeds.csv  ({run_info['minutes']} min)")


if __name__ == "__main__":
    main()
