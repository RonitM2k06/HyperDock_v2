"""
Figures for the manuscript, from saved outputs only.

Inputs
  --metrics  per_run_metrics.csv from experiments/matched_budget_analysis.py (full run)
  --fronts   fronts_full_matched.npz (feasible non-dominated fronts per run)
  --baselines  directory with runs/ of experiments/heuristic_baselines.py
Outputs (PDF and PNG) in --out:
  fig_feasibility   feasible-run rate per dataset and method
  fig_hv            hypervolume per run (NSGA-II vs variant), BFD-V as a marker
  fig_fronts        pooled fronts of NSGA-II, the variant and BFD-V on two datasets
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

C = {"nsga2": "#2a78d6", "nsga2_repair": "#eb6834", "nsga2_seeded": "#1baf7a",
     "nsga2_repair_large": "#eda100", "random": "#8a8a85", "BFD-V": "#2b2b29"}
LABEL = {"nsga2": "NSGA-II", "nsga2_repair": "Volume repair", "nsga2_seeded": "Seeded NSGA-II",
         "nsga2_repair_large": "Large-item repair", "random": "Random search", "BFD-V": "BFD-V (deterministic)"}
DS = [f"{s}-{n}" for s in ("SC1", "SC2") for n in (50, 100, 250, 500)]
INK, MUTED = "#2b2b29", "#6b6b66"

plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False,
                     "axes.spines.right": False, "legend.frameon": False})


def save(fig, out, name):
    for ext in ("pdf", "png"):
        fig.savefig(out / f"{name}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)


def m_df_methods(m):
    return m.method.unique()


def fig_feasibility(m, out):
    present = set(m.method.unique())
    methods = [k for k in ["nsga2", "nsga2_repair", "nsga2_seeded", "nsga2_repair_large", "BFD-V"]
               if k in present]
    rate = m.groupby(["dataset", "method"])["feasible"].mean().unstack()
    fig, ax = plt.subplots(figsize=(7.0, 2.6))
    w = 0.8 / len(methods)
    x = np.arange(len(DS))
    for k, meth in enumerate(methods):
        vals = rate.reindex(DS)[meth].values
        ax.bar(x + (k - (len(methods) - 1) / 2) * w, vals, w * 0.88, color=C[meth], label=LABEL[meth])
    ax.set_xticks(x, DS)
    ax.set_ylabel("Feasible-run rate")
    ax.set_ylim(0, 1.05)
    ax.yaxis.grid(True, color="#e6e5e0", lw=0.6)
    ax.set_axisbelow(True)
    ax.legend(ncol=5, loc="upper center", bbox_to_anchor=(0.5, 1.18), fontsize=8)
    save(fig, out, "fig_feasibility")


def fig_hv(m, out):
    fig, ax = plt.subplots(figsize=(7.0, 2.8))
    rng = np.random.default_rng(0)
    for i, ds in enumerate(DS):
        for k, meth in enumerate(["nsga2", "nsga2_repair", "nsga2_seeded"]):
            v = m[(m.dataset == ds) & (m.method == meth)]["HV"].values
            if len(v) == 0:
                continue
            xpos = i + (k - 1) * 0.27
            ax.boxplot(v, positions=[xpos], widths=0.2, showfliers=False,
                       medianprops=dict(color=INK, lw=1.2),
                       boxprops=dict(color=C[meth]), whiskerprops=dict(color=C[meth]),
                       capprops=dict(color=C[meth]))
            ax.scatter(xpos + rng.uniform(-0.07, 0.07, len(v)), v, s=8, color=C[meth], alpha=0.6,
                       lw=0, label=LABEL[meth] if i == 0 else None)
        b = m[(m.dataset == ds) & (m.method == "BFD-V")]["HV"].values
        if len(b) and b[0] > 0:
            ax.scatter([i], b, marker="D", s=30, color=C["BFD-V"], edgecolor="white", lw=1,
                       zorder=5, label=LABEL["BFD-V"] if ds == "SC1-100" else None)
    ax.set_xticks(range(len(DS)), DS)
    ax.set_ylabel("Hypervolume (normalised)")
    ax.yaxis.grid(True, color="#e6e5e0", lw=0.6)
    ax.set_axisbelow(True)
    ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.17))
    save(fig, out, "fig_hv")


def fig_fronts(fr, base_dir, out, datasets=("SC2-50", "SC1-250")):
    fig, axes = plt.subplots(1, len(datasets), figsize=(7.0, 2.9))
    for ax, ds in zip(axes, datasets):
        for meth in ["nsga2", "nsga2_repair", "nsga2_seeded"]:
            pts = [fr[k] for k in fr.files if k.startswith(f"{ds}__{meth}__") and k.endswith("__F")]
            if pts:
                P = np.vstack(pts)
                ax.scatter(P[:, 0], -P[:, 1], s=7, color=C[meth], alpha=0.55, lw=0, label=LABEL[meth])
        bj = Path(base_dir) / "runs" / f"{ds}__BFD-V__seed0.json"
        if bj.exists():
            meta = json.loads(bj.read_text())
            if meta["feasible"]:
                ax.scatter([meta["F"][0]], [-meta["F"][1]], marker="D", s=36, color=C["BFD-V"],
                           edgecolor="white", lw=1, zorder=5, label=LABEL["BFD-V"])
        ax.set_title(ds, fontsize=9, color=INK)
        ax.set_xlabel("CoM deviation (m), lower is better")
        ax.grid(True, color="#e6e5e0", lw=0.6)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("Mean CASC, higher is better")
    handles, labels = [], []
    for ax in axes:
        for h, l in zip(*ax.get_legend_handles_labels()):
            if l not in labels:
                handles.append(h); labels.append(l)
    fig.legend(handles, labels, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.06))
    fig.tight_layout()
    save(fig, out, "fig_fronts")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics", required=True)
    ap.add_argument("--fronts", required=True)
    ap.add_argument("--baselines", required=True)
    ap.add_argument("--out", default="docs/paper/figures")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    m = pd.read_csv(a.metrics)
    fig_feasibility(m, out)
    fig_hv(m, out)
    fig_fronts(np.load(a.fronts), a.baselines, out)
    print("figures written to", out)


if __name__ == "__main__":
    main()
