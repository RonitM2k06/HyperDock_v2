"""
Deterministic heuristic baselines in the matched-budget output format.

Each baseline is one construction with the EP decoder and the baseline packing rule
in core/placement/baselines.py (all six rotations tried at each extreme point):
  FF      items in dataset order
  BFD-V   items sorted by descending volume
  BFD-P   items sorted by descending priority field

A baseline performs exactly one construction, so it is recorded as one run (seed 0,
n_eval = 1). Its objectives use the same code as CargoProblem (CoM and CASC), and its
constraints are the unplaced volume and count, so experiments/matched_budget_analysis.py
treats it like any other method. The analysis pools baseline points into the
normalisation and reference front of each dataset, as the protocol specifies.

Note: the baselines' rotation rule differs from the NSGA-II decode, which fixes one
orientation per item. Both rules are reported as defined; they are not identical.

Outputs (same layout as matched_budget_runner.py):
  <out>/runs/<dataset>__<method>__seed0.json
  <out>/raw/<dataset>__<method>__seed0.npz      (F, G with one row each)

Usage (repository root):
  python experiments/heuristic_baselines.py --out experiments/full_matched
"""
import argparse
import json
import sys
import time
import platform
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.placement.baselines import (                          # noqa: E402
    FirstFitPacker, BestFitDecreasingVolumePacker, BestFitDecreasingPriorityPacker)
from optimization.nsga2.problem import CargoProblem            # noqa: E402

DATASET_DIR = ROOT / "datasets" / "hyperdock_bench"
METHODS = {
    "FF": FirstFitPacker,
    "BFD-V": BestFitDecreasingVolumePacker,
    "BFD-P": BestFitDecreasingPriorityPacker,
}
ALL_DATASETS = [f"{s}-{n}" for s in ("SC1", "SC2", "SC3") for n in (50, 100, 250, 500)]


def volume(dims):
    return float(dims[0]) * float(dims[1]) * float(dims[2])


def run_one(dataset, method, out_dir):
    path = DATASET_DIR / f"HB-V1-{dataset}.json"
    data = json.loads(path.read_text())
    items = data["items"]
    container = data["containers"][0]["dimensions_wdh_meters"]
    n = len(items)

    t0 = time.perf_counter()
    packer = METHODS[method](container)
    packer.pack(items)
    packed = packer.decoder.packed_items
    runtime = time.perf_counter() - t0

    total_vol = sum(volume(it["dimensions_wdh_meters"]) for it in items)
    packed_vol = sum(volume(p["dimensions"]) for p in packed)
    g1 = max(0.0, total_vol - packed_vol)
    g2 = float(n - len(packed))

    # Objectives: the same evaluators and empty-packing convention as CargoProblem.
    evaluator = CargoProblem(items=items, container_dims=container)
    if len(packed) == 0:
        f1, f2 = 9999.0, 0.0
    else:
        f1 = float(evaluator.com_evaluator.evaluate_layout(packed)["delta_com"])
        f2 = -float(evaluator.casc_evaluator.evaluate_layout(packed)["mean_casc"])

    runs_dir = out_dir / "runs"
    raw_dir = out_dir / "raw"
    runs_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{dataset}__{method}__seed0"
    np.savez_compressed(raw_dir / f"{stem}.npz",
                        F=np.array([[f1, f2]]), G=np.array([[g1, g2]]),
                        X=np.zeros((1, 0)), n_items=np.array(n))
    meta = {
        "dataset": dataset, "method": method, "seed": 0, "n_items": n,
        "container_side_m": container[0], "n_eval": 1, "decodes": 1,
        "runtime_s": runtime, "packed_items": len(packed), "unplaced_count": int(g2),
        "unplaced_volume": g1, "feasible": bool(g2 == 0),
        "F": [f1, f2], "G": [g1, g2],
        "python": platform.python_version(), "platform": platform.platform(),
        "dataset_file": str(path.relative_to(ROOT)),
    }
    (runs_dir / f"{stem}.json").write_text(json.dumps(meta, indent=2))
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=ALL_DATASETS)
    ap.add_argument("--out", default="experiments/full_matched")
    args = ap.parse_args()
    out_dir = (ROOT / args.out).resolve()
    for ds in args.datasets:
        for method in METHODS:
            m = run_one(ds, method, out_dir)
            print(f"[baseline] {ds} {method} packed={m['packed_items']}/{m['n_items']} "
                  f"feasible={m['feasible']} F={m['F'][0]:.4f},{m['F'][1]:.4f} "
                  f"time={m['runtime_s']:.2f}s", flush=True)


if __name__ == "__main__":
    main()
