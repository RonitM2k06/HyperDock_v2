"""
Feasibility-conditioned comparison (manuscript Table 4).

For each dataset and pair of methods, restrict to seeds on which BOTH methods have a
feasible point, then compare HV and IGD (paired Wilcoxon) and report the mean of the
per-run best CoM deviation and best mean CASC among feasible points.

Input: per_run_metrics.csv from experiments/matched_budget_analysis.py.
Usage: python experiments/conditional_analysis.py --metrics experiments/full_matched/per_run_metrics.csv
"""
import argparse
import numpy as np
import pandas as pd
from scipy import stats

PAIRS = [("nsga2_repair", "nsga2"), ("nsga2_seeded", "nsga2"), ("nsga2_repair_large", "nsga2"),
         ("nsga2_repair_large", "nsga2_repair"), ("nsga2_repair", "nsga2_seeded"),
         ("faps", "nsga2"), ("faps", "nsga2_repair"), ("faps", "nsga2_seeded")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--min-seeds", type=int, default=6)
    a = ap.parse_args()
    m = pd.read_csv(a.metrics)
    rows = []
    for ds in sorted(m.dataset.unique()):
        for x, y in PAIRS:
            A = m[(m.dataset == ds) & (m.method == x)].set_index("seed")
            B = m[(m.dataset == ds) & (m.method == y)].set_index("seed")
            common = A.index.intersection(B.index)
            both = [s for s in common if A.loc[s, "feasible"] and B.loc[s, "feasible"]]
            if len(both) < a.min_seeds:
                continue
            r = {"dataset": ds, "comparison": f"{x} vs {y}", "seeds_both_feasible": len(both)}
            for k in ("HV", "IGD", "best_com_dev_feasible", "best_casc_feasible"):
                u, v = A.loc[both, k].values, B.loc[both, k].values
                r[f"{k}_{x}"], r[f"{k}_{y}"] = u.mean(), v.mean()
                r[f"{k}_p"] = 1.0 if np.allclose(u, v) else float(stats.wilcoxon(u, v).pvalue)
            rows.append(r)
    out = pd.DataFrame(rows)
    out.to_csv(a.out or a.metrics.replace("per_run_metrics.csv", "conditional_tests.csv"), index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print(out.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
