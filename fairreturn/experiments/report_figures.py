"""Figures sized for the written report (6.4 inches wide, 8-pt text at 100 % zoom) -> outputs/figures/report/.

    python -m fairreturn.experiments.report_figures

make_figures.py draws the same evidence for slides and the web app (larger canvases); this module redraws
it so that text stays legible when a figure spans one A4 text column. Colours keep one meaning everywhere:
country-aware baseline blue, GBDT-Blind violet, Reweighing aqua, our model orange, variants amber.
"""
from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from fairreturn import config as cfg  # noqa: E402

OUT = cfg.FIG_DIR / "report"
W = 6.4
INK, INK2, MUTED, GRID, AXIS = "#1b1b1b", "#4a4a46", "#898781", "#e3e2dc", "#bdbcb2"
BLUE, VIOLET, AQUA, ORANGE, AMBER, GREY = "#2a78d6", "#6250d6", "#1baf7a", "#eb6834", "#d99a00", "#898781"
ROLE = {"baseline": BLUE, "fairness baseline": AQUA, "reference": GREY, "ablation": VIOLET, "ours": ORANGE,
        "variant": AMBER}

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8, "axes.titlesize": 8.5, "axes.titleweight": "bold",
    "axes.titlelocation": "left", "axes.labelsize": 8, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "legend.fontsize": 7.5, "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "xtick.color": INK2,
    "ytick.color": INK2, "text.color": INK, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5, "axes.axisbelow": True,
    "legend.frameon": False, "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
})


def _save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / name, dpi=300, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    print("  wrote", OUT / name)


def _summary():
    f = cfg.RESULTS_DIR / "summary.csv"
    return pd.read_csv(f) if f.exists() else None


# ----------------------------------------------------------------------------- data
def fig_data():
    L = pd.read_pickle(cfg.LINES_FILE)
    L = L[L["split"].isin(["train", "val", "test"])]
    g = L.groupby("group").agg(rate=("returned", "mean"), lines=("returned", "size"),
                               customers=("customer_id", "nunique")).reindex(cfg.GROUPS)
    fig, (a, b) = plt.subplots(1, 2, figsize=(W, 2.35), gridspec_kw={"width_ratios": [1, 1.15]})
    y = np.arange(len(g))
    a.barh(y, g["rate"] * 100, color="#5b6b80", height=0.62)
    a.set_yticks(y, [f"{k}\n{int(r.customers):,} customers" for k, r in g.iterrows()], fontsize=7)
    a.invert_yaxis()
    a.grid(axis="y", visible=False)
    for yi, v in zip(y, g["rate"] * 100):
        a.text(v + 0.06, yi, f"{v:.2f}%", va="center", fontsize=7, color=INK2)
    a.set_xlim(0, g["rate"].max() * 100 * 1.25)
    a.set_xlabel("% of order lines returned within 60 days")
    a.set_title("(a) Return rate by country group")
    L = L.assign(month=L["invoice_date"].dt.to_period("M").dt.to_timestamp(),
                 uk=np.where(L["group"] == "United Kingdom", "United Kingdom", "Other countries"))
    m = L.groupby(["month", "uk"])["returned"].mean().unstack() * 100
    b.plot(m.index, m["United Kingdom"], color=BLUE, lw=1.6)
    b.plot(m.index, m["Other countries"], color=AQUA, lw=1.6)
    b.set_ylim(0, m.to_numpy().max() * 1.18)
    v0, t0 = pd.Timestamp(cfg.SPLIT_DATES["val_start"]), pd.Timestamp(cfg.SPLIT_DATES["test_start"])
    end = m.index.max() + pd.Timedelta(days=15)
    b.axvspan(v0 - pd.Timedelta(days=15), t0 - pd.Timedelta(days=15), color="#f1f0ec", zorder=0, lw=0)
    b.axvspan(t0 - pd.Timedelta(days=15), end, color="#e6e5df", zorder=0, lw=0)
    top = b.get_ylim()[1]
    b.text(v0 + (t0 - v0) / 2 - pd.Timedelta(days=15), top * 0.97, "val.", ha="center", va="top", fontsize=6.5,
           color=INK2)
    b.text(t0 + (end - t0) / 2 - pd.Timedelta(days=8), top * 0.97, "test", ha="center", va="top", fontsize=6.5,
           color=INK2)
    peak = m["Other countries"].idxmax()
    b.text(peak + pd.Timedelta(days=20), m["Other countries"].max(), "other countries", color=AQUA, fontsize=7,
           va="center", weight="bold")
    b.text(m.index[1], m["United Kingdom"].min() * 0.62, "United Kingdom", color=BLUE, fontsize=7, weight="bold")
    import matplotlib.dates as mdates
    b.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 7]))
    b.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    b.set_ylabel("% returned")
    b.set_title("(b) Monthly return rate")
    fig.tight_layout(w_pad=1.2)
    _save(fig, "r_data.png")


# ----------------------------------------------------------------------------- diagrams
DIAGRAMS = cfg.ROOT / "report" / "diagrams"


def fig_diagrams():
    """Figures 2 and 3 are Graphviz drawings (report/diagrams/*.dot); Graphviz sizes every box to its text."""
    import shutil
    import subprocess
    if shutil.which("dot") is None:
        return print("  skip diagrams: Graphviz 'dot' not installed (apt install graphviz); PNGs in report/diagrams")
    OUT.mkdir(parents=True, exist_ok=True)
    for src, name in (("architecture.dot", "r_architecture.png"), ("system.dot", "r_system.png")):
        subprocess.run(["dot", "-Tpng", str(DIAGRAMS / src), "-o", str(OUT / name)], check=True)
        print("  wrote", OUT / name)


# ----------------------------------------------------------------------------- results
def fig_fpr():
    g = pd.read_csv(cfg.RESULTS_DIR / "group_metrics.csv")
    order = [("gbdt_aware", BLUE), ("gbdt_blind", VIOLET), ("gbdt_reweigh", AQUA), ("vib_grl", ORANGE)]
    order = [(m, c) for m, c in order if m in set(g["method"])]
    groups = [x for x in cfg.GROUPS if x != "EIRE"]
    fig, ax = plt.subplots(figsize=(W, 2.45))
    x = np.arange(len(groups))
    w = 0.8 / len(order)
    for k, (m, color) in enumerate(order):
        sub = g[g["method"] == m].set_index("group").reindex(groups)
        vals = sub["fpr"].to_numpy() * 100
        xs = x + (k - (len(order) - 1) / 2) * w
        ax.bar(xs, vals, width=w * 0.9, color=color, label=sub["label"].dropna().iloc[0])
        for xi, v in zip(xs, vals):
            ax.text(xi, v + 0.35, f"{v:.0f}", ha="center", fontsize=6.5, color=INK2)
    ax.set_xticks(x, groups)
    ax.set_ylabel("false-alarm rate (%)")
    ax.grid(axis="x", visible=False)
    ax.set_ylim(0, ax.get_ylim()[1] * 1.05)
    ax.legend(ncol=2, loc="lower left", bbox_to_anchor=(0, 1.0), borderaxespad=0.2, handlelength=1.2,
              columnspacing=1.5)
    _save(fig, "r_fpr.png")


def fig_postage():
    rows = json.loads((cfg.RESULTS_DIR / "analysis.json").read_text(encoding="utf-8")).get("postage_effect") or []
    labels = [r["label"].replace(" (country column removed)", "").replace(" (country-aware)", ", country-aware")
              .replace("-Unaware", ", country removed") for r in rows]
    vals = np.array([r["mean_delta_logit"] for r in rows])
    fig, ax = plt.subplots(figsize=(W, 0.32 * len(rows) + 0.6))
    yy = np.arange(len(rows))
    ax.barh(yy, vals, color=[ROLE.get(r["role"], GREY) for r in rows], height=0.6)
    ax.axvline(0, color=INK2, lw=0.8)
    ax.set_yticks(yy, labels)
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    span = max(0.05, np.abs(vals).max())
    for yi, v, r in zip(yy, vals, rows):
        ax.text(v + (0.02 if v >= 0 else -0.02), yi, f"{v:+.2f}  ({r['share_lowered']:.0%} of lines lowered)",
                va="center", ha="left" if v >= 0 else "right", fontsize=6.8, color=INK2)
    ax.set_xlim(min(0.0, vals.min()) - 0.62 * span / 0.48, max(0.0, vals.max()) + 0.62 * span / 0.48)
    ax.set_xlabel("mean change in log-odds of a return when only a postage line is added")
    _save(fig, "r_postage.png")


def fig_sweep():
    f = cfg.RESULTS_DIR / "sweep" / "sweep.csv"
    if not f.exists():
        return print("  skip sweep")
    s = pd.read_csv(f)
    lams = sorted(s["lambda_max"].unique())
    pos = {v: i for i, v in enumerate(lams)}
    style = {0.0: (GREY, "o", "β = 0 (no bottleneck)"), 1e-3: (ORANGE, "s", "β = 0.001 (default)"),
             1e-2: ("#8a3412", "^", "β = 0.01")}
    fig, axes = plt.subplots(1, 3, figsize=(W, 2.0))
    panels = (("auc", "(a) AUC ↑"), ("fpr_gap", "(b) ΔFPR ↓"), ("probe_best", "(c) z probe, balanced acc. ↓"))
    for ax, (col, title) in zip(axes, panels):
        for beta, sub in s.groupby("beta"):
            color, marker, label = style.get(beta, (INK2, "o", f"β = {beta:g}"))
            sub = sub.sort_values("lambda_max")
            ax.plot([pos[v] for v in sub["lambda_max"]], sub[col], color=color, marker=marker, ms=4, lw=1.3,
                    label=label)
        d = s[(s["beta"] == 1e-3) & (s["lambda_max"] == 1.0)]
        if len(d):
            ax.scatter([pos[1.0]], d[col], s=90, facecolor="none", edgecolor=INK, linewidth=0.8, zorder=4)
        ax.set_xticks(range(len(lams)), [f"{v:g}" for v in lams])
        ax.set_xlabel("λ_max (adversary weight)")
        ax.set_title(title)
    axes[1].legend(loc="upper right", fontsize=6.3, handlelength=1.6)
    fig.tight_layout(w_pad=1.0)
    _save(fig, "r_sweep.png")


def fig_training():
    f = cfg.MODELS_DIR / "vib_grl" / "info.json"
    h = json.loads(f.read_text(encoding="utf-8")).get("history", {})
    ep = h["epoch"]
    fig, axes = plt.subplots(1, 4, figsize=(W, 1.75))
    axes[0].plot(ep, h["loss_y"], color=ORANGE, lw=1.4)
    axes[0].set_title("return loss")
    axes[1].plot(ep, h["loss_adv"], color=BLUE, lw=1.4)
    axes[1].axhline(np.log(len(cfg.GROUPS)), color=INK2, ls="--", lw=0.7)
    axes[1].text(ep[-1], np.log(len(cfg.GROUPS)) - 0.004, "chance (ln 6)", ha="right", va="top", fontsize=6.3,
                 color=INK2)
    axes[1].set_title("adversary loss")
    axes[2].plot(ep, h["val_auc"], color=ORANGE, lw=1.4)
    axes[2].set_title("validation AUC")
    if h.get("val_fpr_gap"):
        axes[3].plot(ep, h["val_fpr_gap"], color=ORANGE, lw=1.4, label="ΔFPR")
        axes[3].plot(ep, h["val_tpr_gap"], color=VIOLET, lw=1.4, label="ΔTPR")
        axes[3].legend(fontsize=6.5, loc="upper right")
    axes[3].set_title("validation gaps")
    for ax in axes:
        ax.set_xlabel("epoch")
        ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True, nbins=4))
        if h.get("selected_epoch"):
            ax.axvline(h["selected_epoch"], color=AXIS, ls="--", lw=0.7, zorder=0)
    fig.tight_layout(w_pad=0.8)
    _save(fig, "r_training.png")


def fig_leakage():
    s = _summary()
    inp = json.loads((cfg.RESULTS_DIR / "input_probe.json").read_text(encoding="utf-8"))
    names = {"aware": "inputs incl. country", "unaware": "inputs, country removed", "blind": "inputs, country + postage removed"}
    rows = [(names.get(k, k), v["probe_best"], 0.0, "reference") for k, v in inp.items()]
    for _, r in s.iterrows():
        if pd.notna(r.get("probe_best_mean")):
            rows.append((f"z of {r['label']}", r["probe_best_mean"], r["probe_best_std"], r["role"]))
    fig, ax = plt.subplots(figsize=(W, 0.24 * len(rows) + 0.55))
    yy = np.arange(len(rows))
    ax.barh(yy, [r[1] for r in rows], xerr=[r[2] for r in rows], color=[ROLE.get(r[3], GREY) for r in rows],
            height=0.62, error_kw={"elinewidth": 0.6, "ecolor": INK2})
    ax.set_yticks(yy, [r[0] for r in rows], fontsize=7)
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    for yi, r in zip(yy, rows):
        ax.text(r[1] + r[2] + 0.01, yi, f"{r[1]:.2f}", va="center", fontsize=6.8, color=INK2)
    ax.axvline(1 / len(cfg.GROUPS), color=INK2, ls="--", lw=0.8)
    ax.set_ylim(len(rows) - 0.5, -1.25)
    ax.text(1 / len(cfg.GROUPS) + 0.008, -0.85, "chance = 1/6 = 0.17", fontsize=6.5, color=INK2, va="center")
    ax.set_xlim(0, 1.08)
    ax.set_xlabel("balanced accuracy of a fresh classifier recovering the country group")
    _save(fig, "r_leakage.png")


def fig_tsne():
    from sklearn.manifold import TSNE
    from fairreturn.models.registry import LoadedModel
    L = pd.read_pickle(cfg.LINES_FILE)
    te = L[L["split"] == "test"].copy()
    te["g3"] = np.where(te["group"].isin(["United Kingdom", "Germany"]), te["group"], "Other countries")
    rng = np.random.default_rng(0)
    cols = {"United Kingdom": BLUE, "Germany": ORANGE, "Other countries": AQUA}
    idx = np.concatenate([rng.choice(np.flatnonzero(te["g3"].to_numpy() == k), size=min(600, (te["g3"] == k).sum()),
                                     replace=False) for k in cols])
    sample = te.iloc[idx]
    fig, axes = plt.subplots(1, 2, figsize=(W, 2.6))
    for ax, n in zip(axes, ("mlp", "vib_grl")):
        lm = LoadedModel(n)
        xy = TSNE(n_components=2, random_state=0, init="pca", perplexity=35).fit_transform(lm.model.encode(lm.matrix(sample)))
        for k, c in cols.items():
            m = sample["g3"].to_numpy() == k
            ax.scatter(xy[m, 0], xy[m, 1], s=3, alpha=0.6, color=c, edgecolor="none", label=k)
        ax.set_title(lm.label)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
    axes[1].legend(markerscale=3, loc="lower right", fontsize=6.5)
    fig.tight_layout()
    _save(fig, "r_tsne.png")


def fig_stress():
    import glob
    files = sorted(glob.glob(str(cfg.RESULTS_DIR / "stress_bias_*" / "summary.csv")))
    main = _summary()
    if not files or main is None:
        return print("  skip stress")
    s = pd.read_csv(files[-1]).set_index("method")
    m = main.set_index("method")
    keep = [k for k in ("gbdt_blind", "gbdt_reweigh", "mlp", "vib_grl") if k in s.index and k in m.index]
    fig, ax = plt.subplots(figsize=(W, 1.9))
    x = np.arange(len(keep))
    colors = [ROLE[m.loc[k, "role"]] for k in keep]
    ax.bar(x - 0.2, m.loc[keep, "fpr_gap_mean"], 0.38, color=colors, alpha=0.38)
    ax.bar(x + 0.2, s.loc[keep, "fpr_gap_mean"], 0.38, color=colors)
    for xi, (a_, b_) in enumerate(zip(m.loc[keep, "fpr_gap_mean"], s.loc[keep, "fpr_gap_mean"])):
        ax.text(xi - 0.2, a_ + 0.002, f"{a_:.3f}", ha="center", fontsize=6.5, color=INK2)
        ax.text(xi + 0.2, b_ + 0.002, f"{b_:.3f}", ha="center", fontsize=6.5, color=INK2)
    ax.set_xticks(x, [m.loc[k, "label"].split(" (")[0] for k in keep])
    ax.set_ylabel("ΔFPR on clean test labels")
    ax.grid(axis="x", visible=False)
    ax.set_ylim(0, ax.get_ylim()[1] * 1.08)
    ax.legend(handles=[Patch(color=INK2, alpha=0.38, label="trained on clean labels"),
                       Patch(color=INK2, label="trained with 3% label bias injected")], ncol=2,
              loc="lower left", bbox_to_anchor=(0, 1.0), borderaxespad=0.2)
    _save(fig, "r_stress.png")


def main():
    print("Report figures ...")
    for fn in (fig_data, fig_diagrams, fig_fpr, fig_postage, fig_sweep, fig_training, fig_leakage,
               fig_tsne, fig_stress):
        try:
            fn()
        except Exception as e:  # noqa: BLE001
            print(f"  [warn] {fn.__name__} failed: {e}")


if __name__ == "__main__":
    main()
