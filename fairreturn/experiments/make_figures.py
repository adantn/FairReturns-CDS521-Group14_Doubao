"""Report / slide figures from the files written by prepare, run_all, product_tagger and llm_audit.

    python -m fairreturn.experiments.make_figures

Writes PNGs (200 dpi) to outputs/figures/. A figure whose inputs do not exist yet is skipped.
Colours follow one fixed role mapping in every figure (baseline blue, fairness baseline aqua, ablation
violet, ours orange, reference grey); the categorical palette was checked for colour-blind separation.
"""
from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from fairreturn import config as cfg  # noqa: E402

INK, INK2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
ROLE_COLOR = {"baseline": "#2a78d6", "fairness baseline": "#1baf7a", "reference": "#898781",
              "ablation": "#6250d6", "ours": "#eb6834", "variant": "#eda100"}
GROUP3 = {"United Kingdom": "#2a78d6", "Germany": "#eb6834", "Other countries": "#1baf7a"}

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.family": "DejaVu Sans", "font.size": 9.5, "axes.titlesize": 11, "axes.titleweight": "bold",
    "axes.titlelocation": "left", "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "xtick.color": INK2,
    "ytick.color": INK2, "text.color": INK, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "legend.frameon": False,
})


def _save(fig, name: str):
    cfg.FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(cfg.FIG_DIR / name, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("  wrote", cfg.FIG_DIR / name)


def _load_summary():
    f = cfg.RESULTS_DIR / "summary.csv"
    return pd.read_csv(f) if f.exists() else None


def _hbar(ax, labels, values, errs, colors, fmt="{:.3f}", xlim=None):
    y = np.arange(len(labels))
    errs = np.nan_to_num(np.asarray(errs, dtype=float))
    ax.barh(y, values, xerr=errs if np.any(errs > 0) else None, color=colors, height=0.62, edgecolor=SURFACE,
            linewidth=1.5, error_kw={"ecolor": INK2, "elinewidth": 0.8, "capsize": 2})
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    span = (xlim[1] - xlim[0]) if xlim else (np.nanmax(values) if len(values) else 1)
    for yi, v, e in zip(y, values, errs):
        if np.isfinite(v):
            ax.text(v + (e if np.isfinite(e) else 0) + span * 0.01, yi, fmt.format(v), va="center",
                    fontsize=8, color=INK2)
    if xlim:
        ax.set_xlim(*xlim)


def _role_legend(fig, roles):
    from matplotlib.patches import Patch
    handles = [Patch(color=ROLE_COLOR[r], label=r) for r in ROLE_COLOR if r in roles]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), fontsize=8.5, bbox_to_anchor=(0.5, -0.02))


# ----------------------------------------------------------------------------- figures
def fig_data_overview():
    if not cfg.LINES_FILE.exists():
        return print("  skip data overview: run fairreturn.data.prepare")
    L = pd.read_pickle(cfg.LINES_FILE)
    L = L[L["split"].isin(["train", "val", "test"])]
    g = L.groupby("group").agg(rate=("returned", "mean"), lines=("returned", "size"),
                               customers=("customer_id", "nunique")).reindex(cfg.GROUPS)
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 3.8), gridspec_kw={"width_ratios": [1, 1.25]})
    ax = axes[0]
    colors = [ROLE_COLOR["baseline"] if gg == "United Kingdom" else ROLE_COLOR["ours"] for gg in g.index]
    _hbar(ax, [f"{gg}\n{int(r.lines):,} lines · {int(r.customers):,} customers" for gg, r in g.iterrows()],
          g["rate"].to_numpy() * 100, np.zeros(len(g)), colors, fmt="{:.2f}%")
    ax.set_title("Return rate by country group")
    ax.set_xlabel("% of purchased lines returned within 60 days")
    ax = axes[1]
    L = L.assign(month=L["invoice_date"].dt.to_period("M").dt.to_timestamp(),
                 uk=np.where(L["group"] == "United Kingdom", "United Kingdom", "Other countries"))
    m = L.groupby(["month", "uk"])["returned"].mean().unstack() * 100
    for col, color in (("United Kingdom", ROLE_COLOR["baseline"]), ("Other countries", ROLE_COLOR["ours"])):
        if col in m:
            ax.plot(m.index, m[col], color=color, linewidth=2, label=col)
    for name, start in (("val", cfg.SPLIT_DATES["val_start"]), ("test", cfg.SPLIT_DATES["test_start"])):
        ax.axvline(pd.Timestamp(start), color=AXIS, linewidth=1, linestyle="--")
        ax.text(pd.Timestamp(start), ax.get_ylim()[0], f" {name}", color=MUTED, fontsize=8, va="bottom")
    import matplotlib.dates as mdates
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10]))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.set_title("Monthly return rate (train → val → test)")
    ax.set_ylabel("% of lines returned")
    ax.legend(loc="upper right", fontsize=8)
    _save(fig, "fig_data_overview.png")


def fig_accuracy_fairness():
    s = _load_summary()
    if s is None:
        return print("  skip accuracy/fairness: run experiments.run_all")
    s = s[s["method"] != "gbdt_notags"]
    colors = [ROLE_COLOR.get(r, MUTED) for r in s["role"]]
    labels = s["label"].tolist()
    fig, axes = plt.subplots(1, 4, figsize=(15, 0.42 * len(s) + 1.6), sharey=True)
    specs = [("auc", "AUC on test ↑", "{:.3f}", None), ("fpr_gap", "False-alarm gap ΔFPR ↓", "{:.3f}", None),
             ("tpr_gap", "Hit-rate gap ΔTPR ↓", "{:.3f}", None),
             ("cf_flip_rate", "Counterfactual flip rate ↓", "{:.2%}", None)]
    for ax, (col, title, f, xl) in zip(axes, specs):
        vals = s[f"{col}_mean"].to_numpy()
        errs = s[f"{col}_std"].fillna(0).to_numpy()
        lo = 0.5 if col == "auc" else 0.0
        hi = (np.nanmax(vals + errs) * 1.25) if col != "auc" else max(0.8, np.nanmax(vals) + 0.03)
        _hbar(ax, labels, vals, errs, colors, fmt=f, xlim=(lo, hi))
        ax.set_title(title, fontsize=10)
        if col == "cf_flip_rate":
            ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    _role_legend(fig, set(s["role"]))
    fig.subplots_adjust(bottom=0.12)
    _save(fig, "fig_accuracy_fairness.png")


def fig_group_rates():
    f = cfg.RESULTS_DIR / "group_metrics.csv"
    if not f.exists():
        return print("  skip group rates: run experiments.run_all")
    g = pd.read_csv(f)
    # one colour per model (the same as in the web app), fixed order
    order = [("gbdt_aware", ROLE_COLOR["baseline"]), ("gbdt_blind", ROLE_COLOR["ablation"]),
             ("gbdt_reweigh", ROLE_COLOR["fairness baseline"]), ("vib_grl", ROLE_COLOR["ours"])]
    order = [(m, c) for m, c in order if m in set(g["method"])]
    groups = [x for x in cfg.GROUPS if x != "EIRE"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.9))
    w = 0.8 / len(order)
    for ax, col, title in ((axes[0], "fpr", "False-alarm rate: non-returned lines flagged high-risk"),
                           (axes[1], "flag_rate", "Share of all lines flagged high-risk")):
        x = np.arange(len(groups))
        for k, (m, color) in enumerate(order):
            sub = g[g["method"] == m].set_index("group").reindex(groups)
            vals = sub[col].to_numpy() * 100
            ax.bar(x + (k - (len(order) - 1) / 2) * w, vals, width=w * 0.92, color=color,
                   edgecolor=SURFACE, linewidth=1, label=sub["label"].dropna().iloc[0] if sub["label"].notna().any() else m)
            for xi, v in zip(x, vals):
                if np.isfinite(v):
                    ax.text(xi + (k - (len(order) - 1) / 2) * w, v + 0.4, f"{v:.0f}", ha="center", fontsize=6.5,
                            color=INK2)
        ax.set_xticks(x, [cfg.GROUP_SHORT.get(gg, gg) for gg in groups])
        ax.set_ylabel("%")
        ax.set_title(title, fontsize=10)
        ax.grid(axis="x", visible=False)
    axes[0].legend(fontsize=7.5, loc="upper right")
    fig.text(0.01, -0.02, "EIRE is left out: its test lines come from 3 customers. Threshold: top 10% of validation "
             "scores.", fontsize=7.5, color=MUTED)
    _save(fig, "fig_group_rates.png")


def fig_leakage():
    s = _load_summary()
    f = cfg.RESULTS_DIR / "input_probe.json"
    if s is None or not f.exists():
        return print("  skip leakage: run experiments.run_all (full, all methods)")
    inp = json.loads(f.read_text(encoding="utf-8"))
    rows = [(f"Input features: {k}", v["probe_best"], 0.0, "reference") for k, v in inp.items()]
    for _, r in s.iterrows():
        if pd.notna(r.get("probe_best_mean")):
            rows.append((f"Representation z: {r['label']}", r["probe_best_mean"], r["probe_best_std"], r["role"]))
    chance = 1 / len(cfg.GROUPS)
    fig, ax = plt.subplots(figsize=(8.5, 0.42 * len(rows) + 1.3))
    _hbar(ax, [r[0] for r in rows], np.array([r[1] for r in rows]), np.nan_to_num(np.array([r[2] for r in rows])),
          [ROLE_COLOR.get(r[3], MUTED) for r in rows], fmt="{:.2f}", xlim=(0, max(0.75, max(r[1] for r in rows) + 0.1)))
    ax.axvline(chance, color=INK2, linewidth=1, linestyle="--")
    ax.text(chance, len(rows) - 0.45, f" chance {chance:.2f}", color=INK2, fontsize=8, va="center")
    ax.set_title("How well a fresh classifier recovers the country group (balanced accuracy)")
    ax.set_xlabel("probe balanced accuracy  (lower = less country information)")
    _save(fig, "fig_leakage.png")


def fig_pareto():
    f = cfg.RESULTS_DIR / "sweep" / "sweep.csv"
    if not f.exists():
        return print("  skip pareto: run `experiments.run_all --sweep`")
    s = pd.read_csv(f)
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.2))
    markers = {0.0: "o", 1e-3: "s", 1e-2: "^"}
    xlabels = {"fpr_gap": "ΔFPR ↓", "tpr_gap": "ΔTPR ↓", "probe_best": "z probe, balanced accuracy ↓"}
    for ax, col, title in ((axes[0], "fpr_gap", "Accuracy vs false-alarm gap ΔFPR"),
                           (axes[1], "tpr_gap", "Accuracy vs hit-rate gap ΔTPR"),
                           (axes[2], "probe_best", "Accuracy vs country left in z (probe)")):
        for beta, sub in s.groupby("beta"):
            sub = sub.sort_values("lambda_max")
            ax.plot(sub[col], sub["auc"], color=AXIS, linewidth=1, zorder=1)
            sc = ax.scatter(sub[col], sub["auc"], c=sub["lambda_max"], cmap="Oranges", vmin=-1, vmax=4,
                            marker=markers.get(beta, "o"), s=55, edgecolor=INK2, linewidth=0.6, zorder=2,
                            label=f"β = {beta:g}")
            for _, r in sub.iterrows():
                ax.annotate(f"λ={r['lambda_max']:g}", (r[col], r["auc"]), fontsize=6.5, color=INK2,
                            xytext=(4, 3), textcoords="offset points")
        d = s[(s["beta"] == 1e-3) & (s["lambda_max"] == 1.0)]
        if len(d):
            ax.scatter(d[col], d["auc"], s=210, facecolor="none", edgecolor=INK, linewidth=1.2, zorder=3)
            ax.annotate("default", (d[col].iloc[0], d["auc"].iloc[0]), fontsize=7.5, color=INK, ha="right",
                        xytext=(-10, 6), textcoords="offset points")
        ax.set_xlabel(xlabels[col])
        ax.set_ylabel("AUC ↑")
        ax.set_title(title, fontsize=10)
    axes[0].legend(fontsize=8, loc="lower right")
    fig.text(0.01, -0.03, "One seed per point; darker = stronger adversary (λ). Sensitivity analysis on the test "
             "period - the default (β = 0.001, λ = 1) was fixed before this sweep.", fontsize=7.5, color=MUTED)
    _save(fig, "fig_pareto.png")


def fig_tsne():
    from fairreturn.data.features import FeatureBuilder  # noqa: F401
    from fairreturn.models.registry import LoadedModel
    names = [n for n in ("mlp", "vib_grl") if (cfg.MODELS_DIR / n).exists()]
    if not names or not cfg.LINES_FILE.exists():
        return print("  skip t-SNE: needs trained neural models")
    from sklearn.manifold import TSNE
    L = pd.read_pickle(cfg.LINES_FILE)
    te = L[L["split"] == "test"].copy()
    te["g3"] = np.where(te["group"].isin(["United Kingdom", "Germany"]), te["group"], "Other countries")
    rng = np.random.default_rng(0)
    idx = np.concatenate([rng.choice(np.flatnonzero(te["g3"].to_numpy() == g), size=min(700, (te["g3"] == g).sum()),
                                     replace=False) for g in GROUP3])
    sample = te.iloc[idx]
    fig, axes = plt.subplots(1, len(names), figsize=(5.4 * len(names), 4.6))
    axes = np.atleast_1d(axes)
    for ax, n in zip(axes, names):
        lm = LoadedModel(n)
        z = lm.model.encode(lm.matrix(sample))
        xy = TSNE(n_components=2, random_state=0, init="pca", perplexity=35).fit_transform(z)
        for g, color in GROUP3.items():
            k = sample["g3"].to_numpy() == g
            ax.scatter(xy[k, 0], xy[k, 1], s=7, alpha=0.65, color=color, label=g, edgecolor="none")
        ax.set_title(lm.label, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
    axes[-1].legend(markerscale=2.5, fontsize=8, loc="best")
    fig.suptitle("t-SNE of the learned representation z (test lines, balanced sample)", x=0.01, ha="left",
                 fontweight="bold", fontsize=11)
    _save(fig, "fig_tsne.png")


def fig_training():
    f = cfg.MODELS_DIR / "vib_grl" / "info.json"
    if not f.exists():
        return print("  skip training curves")
    h = json.loads(f.read_text(encoding="utf-8")).get("history", {})
    if not h.get("epoch"):
        return
    has_gaps = bool(h.get("val_fpr_gap"))
    fig, axes = plt.subplots(1, 5 if has_gaps else 4, figsize=(17.5 if has_gaps else 14, 3.1))
    ep = h["epoch"]
    axes[0].plot(ep, h["loss_y"], color=ROLE_COLOR["ours"], linewidth=2)
    axes[0].set_title("Return loss (BCE)", fontsize=10)
    axes[1].plot(ep, h["lambda"], color=MUTED, linewidth=2)
    axes[1].set_title("λ schedule (warm-up, then ramp)", fontsize=10)
    axes[2].plot(ep, h["loss_adv"], color=ROLE_COLOR["baseline"], linewidth=2)
    axes[2].axhline(np.log(len(cfg.GROUPS)), color=INK2, linestyle="--", linewidth=1)
    axes[2].text(ep[0], np.log(len(cfg.GROUPS)), " chance = log(6)", color=INK2, fontsize=8, va="bottom")
    axes[2].set_title("Adversary loss (balanced CE)", fontsize=10)
    axes[3].plot(ep, h["val_auc"], color=ROLE_COLOR["ours"], linewidth=2)
    sel = h.get("selected_epoch")
    axes[3].set_title("Validation AUC", fontsize=10)
    if has_gaps:
        ax = axes[4]
        ax.plot(ep, h["val_fpr_gap"], color=ROLE_COLOR["ours"], linewidth=2)
        ax.plot(ep, h["val_tpr_gap"], color=ROLE_COLOR["ablation"], linewidth=2)
        ax.text(ep[-1], h["val_fpr_gap"][-1], "  ΔFPR", color=INK2, fontsize=8, va="center")
        ax.text(ep[-1], h["val_tpr_gap"][-1], "  ΔTPR", color=INK2, fontsize=8, va="center")
        ax.set_title("Validation fairness gaps ↓", fontsize=10)
        ax.set_xlim(ep[0] - 0.3, ep[-1] + 2.2)
    for ax in axes:
        ax.set_xlabel("epoch")
        ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
        if sel:
            ax.axvline(sel, color=AXIS, linestyle="--", linewidth=1, zorder=0)
    if sel:
        axes[3].text(sel, min(h["val_auc"]), f" selected epoch {sel}", fontsize=8, color=INK2)
    _save(fig, "fig_training.png")


def fig_postage():
    f = cfg.RESULTS_DIR / "analysis.json"
    if not f.exists():
        return print("  skip postage proxy figure: run experiments.analysis")
    rows = json.loads(f.read_text(encoding="utf-8")).get("postage_effect") or []
    if not rows:
        return print("  skip postage proxy figure: no model with the postage line")
    labels = [r["label"] for r in rows]
    vals = np.array([r["mean_delta_logit"] for r in rows])
    colors = [ROLE_COLOR.get(r["role"], MUTED) for r in rows]
    fig, ax = plt.subplots(figsize=(8.5, 0.5 * len(rows) + 1.4))
    yy = np.arange(len(rows))
    ax.barh(yy, vals, color=colors, height=0.6, edgecolor=SURFACE, linewidth=1.5)
    ax.axvline(0, color=INK2, linewidth=1)
    ax.set_yticks(yy, labels)
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    span = max(0.05, np.abs(vals).max())
    for yi, v, r in zip(yy, vals, rows):
        ax.text(v + (span * 0.03 if v >= 0 else -span * 0.03), yi, f"{v:+.2f}  ({r['share_lowered']:.0%} of lines lowered)",
                va="center", ha="left" if v >= 0 else "right", fontsize=7.5, color=INK2)
    ax.set_xlim(-span * 2.1, span * 2.1)
    ax.set_xlabel("mean change in log-odds of a return when only the postage line is added")
    ax.set_title("The postage line is a country proxy: who uses it, and in which direction")
    _save(fig, "fig_postage_proxy.png")


def fig_tags():
    if not cfg.LINES_FILE.exists():
        return
    L = pd.read_pickle(cfg.LINES_FILE)
    tr = L[L["split"] == "train"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), gridspec_kw={"width_ratios": [1, 1.6]})
    flags = []
    for col, name in (("tag_fragile", "fragile"), ("tag_seasonal", "seasonal")):
        r = tr.groupby(col)["returned"].mean() * 100
        flags += [(f"{name}: no", r.get(0.0, np.nan)), (f"{name}: yes", r.get(1.0, np.nan))]
    _hbar(axes[0], [f[0] for f in flags], np.array([f[1] for f in flags]), np.zeros(len(flags)),
          [MUTED, ROLE_COLOR["ours"], MUTED, ROLE_COLOR["ours"]], fmt="{:.2f}%")
    axes[0].set_title("Return rate by product attribute", fontsize=10)
    c = tr.groupby("tag_category")["returned"].agg(["mean", "size"]).sort_values("mean", ascending=False)
    _hbar(axes[1], [f"{k} ({int(v):,})" for k, v in c["size"].items()], c["mean"].to_numpy() * 100,
          np.zeros(len(c)), [ROLE_COLOR["baseline"]] * len(c), fmt="{:.2f}%")
    axes[1].set_title("Return rate by product category (training lines)", fontsize=10)
    src = L["tag_source"].value_counts(normalize=True)
    fig.text(0.01, -0.03, "Tag source: " + ", ".join(f"{k} {v:.0%}" for k, v in src.items()), fontsize=7.5, color=MUTED)
    _save(fig, "fig_product_tags.png")


def fig_llm_audit():
    f = cfg.AUDIT_DIR / "summary.csv"
    if not f.exists():
        return print("  skip LLM audit figure: run fairreturn.genai.llm_audit")
    s = pd.read_csv(f)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.4))
    colors = [ROLE_COLOR["ours"] if c == "tool" else ROLE_COLOR["baseline"] for c in s["condition"]]
    for ax, col, title, f_ in ((axes[0], "cf_flip_rate", "Decision flips when only the country changes ↓", "{:.0%}"),
                               (axes[1], "mean_abs_delta_risk", "Mean |Δ risk| between twins (points) ↓", "{:.1f}"),
                               (axes[2], "auc", "AUC on the audit sample ↑", "{:.2f}")):
        vals = s[col].to_numpy()
        ax.bar(s["condition"], vals, color=colors, edgecolor=SURFACE, width=0.6)
        for x, v in enumerate(vals):
            if np.isfinite(v):
                ax.text(x, v, f_.format(v), ha="center", va="bottom", fontsize=8, color=INK2)
        ax.set_title(title, fontsize=9.5)
        ax.grid(axis="x", visible=False)
        if col == "cf_flip_rate":
            ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    if "is_mock" in s and s["is_mock"].any():
        fig.suptitle("MOCK BACKEND - pipeline test only, do not report", color="#d03b3b", fontweight="bold")
    _save(fig, "fig_llm_audit.png")


def fig_stress():
    import glob
    files = sorted(glob.glob(str(cfg.RESULTS_DIR / "stress_bias_*" / "summary.csv")))
    main = _load_summary()
    if not files or main is None:
        return print("  skip stress test figure: run run_all --inject-bias ...")
    s = pd.read_csv(files[-1])
    keep = [k for k in ("gbdt_blind", "gbdt_reweigh", "mlp", "vib_grl") if k in set(main["method"]) and k in set(s["method"])]
    m = main[main["method"].isin(keep)].set_index("method")
    s = s[s["method"].isin(keep)].set_index("method")
    fig, ax = plt.subplots(figsize=(7.5, 3.3))
    x = np.arange(len(keep))
    colors = [ROLE_COLOR[m.loc[k, "role"]] for k in keep]
    ax.bar(x - 0.2, m.loc[keep, "fpr_gap_mean"], 0.38, color=colors, alpha=0.38, edgecolor=SURFACE)
    ax.bar(x + 0.2, s.loc[keep, "fpr_gap_mean"], 0.38, color=colors, edgecolor=SURFACE)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=INK2, alpha=0.38, label="clean training labels"),
                       Patch(color=INK2, label="with injected label bias")], fontsize=8)
    for xi, (a_, b_) in enumerate(zip(m.loc[keep, "fpr_gap_mean"], s.loc[keep, "fpr_gap_mean"])):
        ax.text(xi - 0.2, a_, f"{a_:.3f}", ha="center", va="bottom", fontsize=7.5, color=INK2)
        ax.text(xi + 0.2, b_, f"{b_:.3f}", ha="center", va="bottom", fontsize=7.5, color=INK2)
    ax.set_xticks(x, [m.loc[k, "label"].split(" (")[0] for k in keep], fontsize=8)
    ax.set_ylabel("ΔFPR on clean test labels")
    rate = float(files[-1].split('stress_bias_')[1].split('/')[0])
    ax.set_title(f"Stress test: {rate:.0%} of non-returned German/French training lines relabelled as returned",
                 fontsize=9.5)
    ax.grid(axis="x", visible=False)
    _save(fig, "fig_stress_test.png")


def main():
    print("Making figures ...")
    for fn in (fig_data_overview, fig_accuracy_fairness, fig_group_rates, fig_leakage, fig_postage, fig_pareto,
               fig_tsne, fig_training, fig_tags, fig_llm_audit, fig_stress):
        try:
            fn()
        except Exception as e:  # one broken figure must not stop the others
            print(f"  [warn] {fn.__name__} failed: {e}")


if __name__ == "__main__":
    main()
