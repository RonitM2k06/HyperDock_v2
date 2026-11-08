"""
Analysis of matched-budget runs written by experiments/matched_budget_runner.py.

Reads only saved outputs (raw/*.npz and runs/*.json). No experiment is re-run here.

Definitions (fixed before looking at results):
  Archive of a run   = every evaluated chromosome of that run (all evaluations,
                       not only the final population), so NSGA-II and random search
                       are compared on the same evaluation set.
  Run front          = non-dominated feasible points (G <= 0) of the archive.
  Reference front    = non-dominated feasible points pooled over all methods and
                       seeds of one dataset.
  Normalisation      = min-max per dataset over the pooled feasible points.
  HV                 = hypervolume in normalised space, reference point (1.1, 1.1).
                       A run with no feasible point has HV = 0.
  IGD                = inverted generational distance to the normalised reference
                       front (lower is better). Undefined (NaN) without feasible points.
  Feasible-run rate  = fraction of runs with at least one feasible evaluation.

Statistics: per dataset, paired comparisons across seeds (same seed index).
Wilcoxon signed-rank (two-sided) and Cliff's delta. Bootstrap 95% CI of the mean HV.
Holm step-down correction across datasets, within each comparison (wilcoxon_p_holm).
Datasets where both compared methods have HV 0 in every run are excluded from the
Holm family (column 'informative'), since no test is possible there.
With few seeds the minimum achievable two-sided Wilcoxon p is 2/2^n; with n = 3 this
is 0.25, so the reduced run cannot reach significance and is reported as descriptive.

Usage:
  python experiments/matched_budget_analysis.py --dir experiments/reduced_sc1
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from pymoo.indicators.hv import HV
from pymoo.indicators.igd import IGD
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

REF_POINT = np.array([1.1, 1.1])
RNG = np.random.default_rng(12345)  # bootstrap seed, fixed for reproducibility


def nondominated(points):
    if len(points) == 0:
        return np.zeros((0, 2))
    idx = NonDominatedSorting().do(points, only_non_dominated_front=True)
    return points[idx]


def cliffs_delta(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) == 0 or len(y) == 0:
        return float("nan")
    return float(np.sign(x[:, None] - y[None, :]).sum() / (len(x) * len(y)))


def bootstrap_ci(values, n_boot=10000, alpha=0.05):
    v = np.asarray(values, float)
    if len(v) == 0:
        return float("nan"), float("nan")
    means = RNG.choice(v, size=(n_boot, len(v)), replace=True).mean(axis=1)
    return float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def load_runs(base: Path):
    runs = []
    for meta_path in sorted((base / "runs").glob("*.json")):
        meta = json.loads(meta_path.read_text())
        raw = np.load(base / "raw" / meta_path.name.replace(".json", ".npz"))
        F = raw["F"]
        G = raw["G"]
        feas = np.all(G <= 0, axis=1)
        runs.append({
            "meta": meta,
            "F": F,
            "feasible_F": F[feas],
            "n_feasible_eval": int(feas.sum()),
            "front": nondominated(F[feas]),
        })
    return runs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="experiments/reduced_sc1")
    args = ap.parse_args()
    base = Path(args.dir)
    runs = load_runs(base)
    if not runs:
        raise SystemExit(f"No runs found in {base}/runs")

    rows = []
    stat_rows = []
    for ds in sorted({r["meta"]["dataset"] for r in runs}):
        ds_runs = [r for r in runs if r["meta"]["dataset"] == ds]
        pooled = np.vstack([r["front"] for r in ds_runs if len(r["front"])] or [np.zeros((0, 2))])
        if len(pooled) == 0:
            # No feasible point in any run of this dataset: every run has HV 0 and IGD undefined.
            # Keep the same columns as the main branch so runtime and decoder counts are not lost.
            for r in ds_runs:
                m = r["meta"]
                rows.append({
                    "dataset": ds, "method": m["method"], "seed": m["seed"],
                    "n_items": m["n_items"], "n_eval": m["n_eval"], "decodes": m["decodes"],
                    "runtime_s": m["runtime_s"], "feasible_evaluations": r["n_feasible_eval"],
                    "feasible_eval_rate": r["n_feasible_eval"] / max(1, len(r["F"])),
                    "feasible_front_size": 0, "feasible": False,
                    "HV": 0.0, "IGD": float("nan"),
                    "best_com_dev_feasible": float("nan"), "best_casc_feasible": float("nan"),
                })
            continue
        ref_front = nondominated(pooled)
        f_min = pooled.min(axis=0)
        f_max = pooled.max(axis=0)
        denom = np.where(f_max - f_min == 0, 1e-12, f_max - f_min)
        norm = lambda P: (P - f_min) / denom  # noqa: E731
        ref_norm = norm(ref_front)
        hv_ind = HV(ref_point=REF_POINT)
        igd_ind = IGD(ref_norm)

        for r in ds_runs:
            m = r["meta"]
            if len(r["front"]):
                fn = norm(r["front"])
                hv = float(hv_ind(fn))
                igd = float(igd_ind(fn))
                best_com = float(r["front"][:, 0].min())
                best_casc = float(-r["front"][:, 1].max())
            else:
                hv, igd, best_com, best_casc = 0.0, float("nan"), float("nan"), float("nan")
            rows.append({
                "dataset": ds, "method": m["method"], "seed": m["seed"],
                "n_items": m["n_items"], "n_eval": m["n_eval"], "decodes": m["decodes"],
                "runtime_s": m["runtime_s"], "feasible_evaluations": r["n_feasible_eval"],
                "feasible_eval_rate": r["n_feasible_eval"] / max(1, len(r["F"])),
                "feasible_front_size": len(r["front"]),
                "feasible": len(r["front"]) > 0,
                "HV": hv, "IGD": igd,
                "best_com_dev_feasible": best_com, "best_casc_feasible": best_casc,
            })

    per_run = pd.DataFrame(rows)
    per_run.to_csv(base / "per_run_metrics.csv", index=False)

    # Aggregates per dataset and method
    agg = []
    for (ds, m), g in per_run.groupby(["dataset", "method"]):
        lo, hi = bootstrap_ci(g["HV"].values)
        agg.append({
            "dataset": ds, "method": m, "runs": len(g),
            "HV_mean": g["HV"].mean(), "HV_std": g["HV"].std(ddof=1),
            "HV_median": g["HV"].median(), "HV_ci95_low": lo, "HV_ci95_high": hi,
            "IGD_mean": g["IGD"].mean(), "IGD_std": g["IGD"].std(ddof=1),
            "feasible_run_rate": g["feasible"].mean(),
            "feasible_eval_rate_mean": g["feasible_eval_rate"].mean(),
            "runtime_mean_s": g["runtime_s"].mean(), "runtime_std_s": g["runtime_s"].std(ddof=1),
            "decodes_mean": g["decodes"].mean(),
        })
    agg_df = pd.DataFrame(agg)
    agg_df.to_csv(base / "aggregate_metrics.csv", index=False)

    # Paired comparisons across seeds: each variant against NSGA-II and random search
    for ds, gd in per_run.groupby("dataset"):
        for a, b in [("nsga2_repair", "nsga2"), ("nsga2_repair", "random"),
                     ("nsga2", "random"), ("nsga2_seeded", "nsga2"),
                     ("nsga2_repair", "nsga2_seeded"), ("nsga2_repair_large", "nsga2"),
                     ("nsga2_repair_large", "nsga2_repair"),
                     ("faps", "nsga2"), ("faps", "nsga2_repair"), ("faps", "nsga2_seeded"),
                     ("sms_repair", "sms_nsga2"), ("sms_seeded", "sms_nsga2"),
                     ("sms_repair", "sms_seeded"), ("sms_faps", "sms_nsga2"),
                     ("sms_faps", "sms_repair"), ("sms_faps", "sms_seeded")]:
            ga = gd[gd.method == a].set_index("seed")["HV"]
            gb = gd[gd.method == b].set_index("seed")["HV"]
            common = ga.index.intersection(gb.index)
            if len(common) == 0:
                continue
            x, y = ga.loc[common].values, gb.loc[common].values
            diffs = x - y
            if np.allclose(diffs, 0):
                p = 1.0
            else:
                try:
                    p = float(stats.wilcoxon(x, y, alternative="two-sided").pvalue)
                except ValueError:
                    p = float("nan")
            stat_rows.append({
                "dataset": ds, "comparison": f"{a} vs {b}", "n_pairs": len(common),
                "mean_diff_HV": float(diffs.mean()), "wilcoxon_p_two_sided": p,
                "cliffs_delta": cliffs_delta(x, y),
                "min_possible_p_two_sided": 2.0 / (2 ** len(common)),
            })
    stat_df = pd.DataFrame(stat_rows)
    # Holm step-down correction across datasets, within each comparison family.
    stat_df["wilcoxon_p_holm"] = np.nan
    # A comparison where both methods have HV 0 in every run (no feasible point at
    # all, e.g. SC3) is not a test; it is excluded from the Holm family and gets NaN.
    if len(stat_df):
        stat_df["informative"] = [
            not (np.allclose(per_run[(per_run.dataset == r.dataset) &
                                     (per_run.method.isin(r.comparison.split(" vs ")))]["HV"], 0))
            for r in stat_df.itertuples()]
    for comp, g in (stat_df.groupby("comparison") if len(stat_df) else []):
        p = g["wilcoxon_p_two_sided"].values
        valid = ~np.isnan(p) & g["informative"].values
        idx = g.index[valid]
        pv = p[valid]
        order = np.argsort(pv)
        m = len(pv)
        adj = np.empty(m)
        running = 0.0
        for rank, k in enumerate(order):
            running = max(running, (m - rank) * pv[k])
            adj[k] = min(1.0, running)
        stat_df.loc[idx, "wilcoxon_p_holm"] = adj
    stat_df.to_csv(base / "paired_tests.csv", index=False)

    print(agg_df.to_string(index=False))
    print()
    print(stat_df.to_string(index=False))


if __name__ == "__main__":
    main()
