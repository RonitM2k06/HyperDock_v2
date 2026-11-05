"""
Sensitivity of the main comparison to the CASC constants and the CoM target.

Input: the feasible non-dominated fronts of every nsga2 and nsga2_repair run of
the full matched-budget run (SC1 and SC2), extracted from experiments/full_matched/raw
into one file (fronts_full_matched.npz, keys '<stem>__X' and '<stem>__F').

For each setting below, every front point is re-decoded once (plain decode for
nsga2, fallback decode for nsga2_repair) and re-scored; BFD-V is re-run. Per
dataset, run fronts are recomputed under the new objective, normalised against
the pooled points, and hypervolume is compared seed-paired (variant vs NSGA-II;
runs without feasible points have HV 0), with Holm correction across datasets.

Limitation: only points on the original front are re-scored. Under a changed
objective, other archive points could join a run's front, so this tests the
robustness of the evaluation, not a re-optimisation.

Usage:
  python experiments/sensitivity_rescore.py --fronts experiments/full_matched/fronts_full_matched.npz \
         --out experiments/sensitivity
"""
import argparse
import json
import sys
from pathlib import Path
from multiprocessing import Pool

import numpy as np
import pandas as pd
from scipy import stats
from pymoo.indicators.hv import HV

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.metrics.casc import CASCv2, CASCv2Config            # noqa: E402
from core.metrics.com import CoMModel, CoMConfig              # noqa: E402
from core.placement.baselines import BestFitDecreasingVolumePacker  # noqa: E402
from optimization.nsga2.feasibility import decode_layout, descending_volume_order  # noqa: E402

DATASETS = [f"{s}-{n}" for s in ("SC1", "SC2") for n in (50, 100, 250, 500)]
SEEDS = {50: 30, 100: 30, 250: 10, 500: 10}
BASE = dict(L_arm=0.75, lambda_occ=1.0, C_grasp=2.0, kappa=0.05)
SETTINGS = {"baseline": {}}
for k in BASE:
    SETTINGS[f"{k}_x0.5"] = {k: BASE[k] * 0.5}
    SETTINGS[f"{k}_x2"] = {k: BASE[k] * 2.0}
SETTINGS["com_target_container_centre"] = {"com_centre": True}


def load(ds):
    d = json.loads((ROOT / "datasets/hyperdock_bench" / f"HB-V1-{ds}.json").read_text())
    return d["items"], d["containers"][0]["dimensions_wdh_meters"]


def scorers(setting, cdims):
    cfg = CASCv2Config(grid_resolution=10, **{k: v for k, v in setting.items() if k in BASE})
    t = np.array([cdims[0] / 2, cdims[1] / 2, cdims[2] / 2]) if setting.get("com_centre") else np.array([0.5, 0.5, 1.0])
    return CASCv2(cfg), CoMModel(CoMConfig(target_com=t, dry_mass=5000.0, dry_com=t))


def decode_job(job):
    ds, method, x = job
    items, cdims = load(ds)
    n = len(items)
    seq, ori = [int(v) for v in x[:n]], x[n:].astype(int)
    uv, uc, packed = decode_layout(items, cdims, seq, ori)
    if uc > 0 and method == "nsga2_repair":
        uv2, uc2, p2 = decode_layout(items, cdims, descending_volume_order(items, seq), ori)
        if uv2 < uv:
            uv, uc, packed = uv2, uc2, p2
    rows = []
    for name, st in SETTINGS.items():
        casc, com = scorers(st, cdims)
        rows.append((name, float(com.evaluate_layout(packed)["delta_com"]),
                     -float(casc.evaluate_layout(packed)["mean_casc"])))
    return uc, rows


def nd(F):
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
    ap.add_argument("--fronts", required=True)
    ap.add_argument("--out", default="experiments/sensitivity")
    ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    fr = np.load(a.fronts)

    jobs, keys = [], []
    for k in fr.files:
        if not k.endswith("__X"):
            continue
        ds, method, seed = k[:-3].split("__")
        for i, x in enumerate(fr[k]):
            jobs.append((ds, method, x))
            keys.append((ds, method, int(seed.replace("seed", "")), i))
    with Pool(a.workers) as pool:
        results = pool.map(decode_job, jobs, chunksize=4)

    pts = []
    bad = 0
    for (ds, method, seed, i), (uc, rows) in zip(keys, results):
        bad += uc > 0
        for name, f1, f2 in rows:
            pts.append((name, ds, method, seed, f1, f2))
    # BFD-V, deterministic
    for ds in DATASETS:
        items, cdims = load(ds)
        pk = BestFitDecreasingVolumePacker(cdims)
        pk.pack(items)
        if len(pk.decoder.packed_items) == len(items):
            for name, st in SETTINGS.items():
                casc, com = scorers(st, cdims)
                pts.append((name, ds, "BFD-V", 0, float(com.evaluate_layout(pk.decoder.packed_items)["delta_com"]),
                            -float(casc.evaluate_layout(pk.decoder.packed_items)["mean_casc"])))
    P = pd.DataFrame(pts, columns=["setting", "dataset", "method", "seed", "f1", "f2"])
    P.to_csv(out / "rescored_points.csv", index=False)

    rows = []
    for name in SETTINGS:
        for ds in DATASETS:
            sub = P[(P.setting == name) & (P.dataset == ds)]
            pooled = sub[["f1", "f2"]].values
            lo, hi = pooled.min(0), pooled.max(0)
            den = np.where(hi - lo == 0, 1e-12, hi - lo)
            hv = HV(ref_point=np.array([1.1, 1.1]))
            hvs = {}
            for (m, s), g in sub.groupby(["method", "seed"]):
                hvs[(m, s)] = float(hv((nd(g[["f1", "f2"]].values) - lo) / den))
            n_seeds = SEEDS[int(ds.split("-")[1])]
            v = np.array([hvs.get(("nsga2_repair", s), 0.0) for s in range(1, n_seeds + 1)])
            b = np.array([hvs.get(("nsga2", s), 0.0) for s in range(1, n_seeds + 1)])
            p = 1.0 if np.allclose(v - b, 0) else float(stats.wilcoxon(v, b).pvalue)
            rows.append({"setting": name, "dataset": ds, "variant_HV_mean": v.mean(), "nsga2_HV_mean": b.mean(),
                         "BFDV_HV": hvs.get(("BFD-V", 0), np.nan), "diff": (v - b).mean(), "p": p})
    R = pd.DataFrame(rows)
    R["p_holm"] = np.nan
    for name, g in R.groupby("setting"):
        R.loc[g.index, "p_holm"] = holm(g["p"].values)
    R.to_csv(out / "sensitivity_results.csv", index=False)

    summ = R.groupby("setting").apply(lambda g: pd.Series({
        "variant_better_datasets": int((g["diff"] > 0).sum()),
        "significant_holm_0.05": int((g["p_holm"] < 0.05).sum()),
        "variant_beats_BFDV": int((g["variant_HV_mean"] > g["BFDV_HV"]).sum()),
        "BFDV_feasible_datasets": int(g["BFDV_HV"].notna().sum())}), include_groups=False)
    summ.to_csv(out / "sensitivity_summary.csv")
    print(f"re-decoded points with unplaced items (should be 0): {bad}")
    print(summ.to_string())


if __name__ == "__main__":
    main()
