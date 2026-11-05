"""
Analysis of experiments/ablation_study.py outputs.

Reference ("full") = the nsga2_repair runs of the full matched-budget run for the
same datasets and seeds, given as their feasible fronts (fronts_full_matched.npz).
Ablation fronts are already re-scored with the full objective.

Per dataset: pool all fronts (full + every mode), min-max normalise, HV with
reference point (1.1, 1.1). Each mode is compared with full, paired by seed
(Wilcoxon two-sided), Holm across modes within a dataset.

Usage:
  python experiments/ablation_analysis.py --ablation experiments/ablation \
         --fronts docs/paper/data/fronts_full_matched.npz
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from pymoo.indicators.hv import HV


def nd(F):
    if len(F) == 0:
        return F
    keep = np.ones(len(F), bool)
    for i in range(len(F)):
        if keep[i] and np.any(np.all(F <= F[i], axis=1) & np.any(F < F[i], axis=1)):
            keep[i] = False
    return F[keep]


def holm(p):
    p = np.asarray(p, float)
    order, m, run, adj = np.argsort(p), len(p), 0.0, np.empty(len(p))
    for r, k in enumerate(order):
        run = max(run, (m - r) * p[k])
        adj[k] = min(1.0, run)
    return adj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ablation", default="experiments/ablation")
    ap.add_argument("--fronts", default="docs/paper/data/fronts_full_matched.npz")
    a = ap.parse_args()
    base = Path(a.ablation)
    fr = np.load(a.fronts)

    fronts, meta = {}, {}
    for j in sorted((base / "runs").glob("*.json")):
        m = json.loads(j.read_text())
        d = np.load(base / "fronts" / j.name.replace(".json", ".npz"))
        F, G = d["F_true"], d["G"]
        ok = np.all(G <= 0, axis=1) if len(G) else np.zeros(0, bool)
        fronts[(m["dataset"], m["mode"], m["seed"])] = nd(F[ok]) if len(F) else np.zeros((0, 2))
        meta[(m["dataset"], m["mode"], m["seed"])] = m
    datasets = sorted({k[0] for k in fronts})
    seeds = sorted({k[2] for k in fronts})
    modes = sorted({k[1] for k in fronts})
    for ds in datasets:
        for s in seeds:
            key = f"{ds}__nsga2_repair__seed{s}__F"
            fronts[(ds, "full", s)] = nd(fr[key]) if key in fr.files else np.zeros((0, 2))

    rows, tests = [], []
    hv = HV(ref_point=np.array([1.1, 1.1]))
    for ds in datasets:
        pooled = np.vstack([f for (d, _, _), f in fronts.items() if d == ds and len(f)])
        lo, hi = pooled.min(0), pooled.max(0)
        den = np.where(hi - lo == 0, 1e-12, hi - lo)
        H = {}
        for mode in ["full"] + modes:
            for s in seeds:
                f = fronts.get((ds, mode, s), np.zeros((0, 2)))
                H[(mode, s)] = float(hv((f - lo) / den)) if len(f) else 0.0
                rows.append({"dataset": ds, "mode": mode, "seed": s, "HV": H[(mode, s)],
                             "feasible": len(f) > 0,
                             "runtime_s": meta.get((ds, mode, s), {}).get("runtime_s", np.nan)})
        full = np.array([H[("full", s)] for s in seeds])
        ds_tests = []
        for mode in modes:
            x = np.array([H[(mode, s)] for s in seeds])
            p = 1.0 if np.allclose(x - full, 0) else float(stats.wilcoxon(x, full).pvalue)
            ds_tests.append({"dataset": ds, "mode": mode, "HV_mean": x.mean(), "HV_sd": x.std(ddof=1),
                             "full_HV_mean": full.mean(), "diff_mean": (x - full).mean(),
                             "feasible_runs": int(sum(H[(mode, s)] > 0 for s in seeds)),
                             "p": p})
        hp = holm([t["p"] for t in ds_tests])
        for t, h in zip(ds_tests, hp):
            t["p_holm"] = h
        tests += ds_tests
    pd.DataFrame(rows).to_csv(base / "ablation_per_run.csv", index=False)
    T = pd.DataFrame(tests)
    T.to_csv(base / "ablation_tests.csv", index=False)
    print(T.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
