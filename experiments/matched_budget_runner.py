"""
Matched-budget comparison runner.

Methods (per dataset and seed):
  nsga2         NSGA-II with CargoProblem (the existing method, unchanged).
  nsga2_repair  NSGA-II with CargoProblemRepair (feasibility-aware fallback decode).
  random        Uniform random sampling with the SAME number of chromosome
                evaluations that the NSGA-II run used for that dataset and seed.

Budget: one unit = one chromosome evaluation (pymoo's evaluator count). The
repair variant may perform up to two decoder calls per evaluation; decoder calls
are recorded separately in the summary and are not hidden in the budget.

Population and generations follow experiments/phase5_runner.py:
  N <= 100 -> population 100, generations 100
  N >= 250 -> population 50,  generations 50

Outputs (one file per run; existing files are skipped so interrupted batches
resume from completed runs):
  <out>/raw/<dataset>__<method>__seed<s>.npz   every evaluated X, F, G
  <out>/runs/<dataset>__<method>__seed<s>.json configuration and measured costs
  <out>/summary.csv                             rebuilt from the JSON files

Usage:
  python experiments/matched_budget_runner.py --datasets SC1-50 SC1-100 SC1-250 SC1-500 \
         --seeds 1 2 3 --out experiments/reduced_sc1
"""
import argparse
import csv
import json
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from optimization.nsga2.problem import CargoProblem            # noqa: E402
from optimization.nsga2.feasibility import CargoProblemRepair  # noqa: E402
from optimization.nsga2.operators import (                     # noqa: E402
    CargoCrossover, CargoMutation, CargoSampling)

DATASET_DIR = ROOT / "datasets" / "hyperdock_bench"
METHODS = ("nsga2", "nsga2_repair", "random")


def pop_and_gen(n_items):
    """Same rule as experiments/phase5_runner.py."""
    return (50, 50) if n_items >= 250 else (100, 100)


def load_dataset(name):
    path = DATASET_DIR / f"HB-V1-{name}.json"
    with open(path, "r") as f:
        data = json.load(f)
    items = data["items"]
    container_dims = data["containers"][0]["dimensions_wdh_meters"]
    return path, items, container_dims


def attach_recorder(problem):
    """Record every evaluated (X, F, G) by wrapping the instance's _evaluate."""
    store = {"X": [], "F": [], "G": []}
    original = problem._evaluate

    def recorded(x, out, *args, **kwargs):
        original(x, out, *args, **kwargs)
        store["X"].append(np.asarray(x, dtype=float).copy())
        store["F"].append(np.asarray(out["F"], dtype=float).copy())
        store["G"].append(np.asarray(out["G"], dtype=float).copy())

    problem._evaluate = recorded
    return store


def run_one(dataset, method, seed, budget_from, out_dir):
    raw_path = out_dir / "raw" / f"{dataset}__{method}__seed{seed}.npz"
    run_path = out_dir / "runs" / f"{dataset}__{method}__seed{seed}.json"
    if run_path.exists() and raw_path.exists():
        return json.loads(run_path.read_text()), "skipped"

    ds_path, items, container_dims = load_dataset(dataset)
    n = len(items)
    pop, gens = pop_and_gen(n)
    np.random.seed(seed)  # pymoo's minimize(seed=...) below also seeds; kept for random sampling

    if method == "nsga2":
        problem = CargoProblem(items=items, container_dims=container_dims)
    elif method == "nsga2_repair":
        problem = CargoProblemRepair(items=items, container_dims=container_dims)
    elif method == "random":
        problem = CargoProblem(items=items, container_dims=container_dims)
    else:
        raise ValueError(method)

    store = attach_recorder(problem)
    t0 = time.perf_counter()

    if method in ("nsga2", "nsga2_repair"):
        algorithm = NSGA2(
            pop_size=pop,
            sampling=CargoSampling(n_items=n),
            crossover=CargoCrossover(n_items=n, prob=0.9),
            mutation=CargoMutation(n_items=n, prob=1.0 / n),
            eliminate_duplicates=True,
        )
        res = minimize(problem, algorithm, ("n_gen", gens), seed=seed, verbose=False)
        # minimize() runs on its own copy of the algorithm, so read the count from the result.
        n_eval = int(res.algorithm.evaluator.n_eval)
    else:
        budget = int(budget_from["n_eval"])
        X = CargoSampling(n_items=n)._do(problem, budget)
        for x in X:
            out = {}
            problem._evaluate(x, out)
        n_eval = budget

    runtime = time.perf_counter() - t0

    raw_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        raw_path,
        X=np.vstack(store["X"]), F=np.vstack(store["F"]), G=np.vstack(store["G"]),
        n_items=np.array(n),
    )

    summary = {
        "dataset": dataset, "method": method, "seed": seed, "n_items": n,
        "container_side_m": container_dims[0],
        "population": pop if method != "random" else None,
        "generations": gens if method != "random" else None,
        "n_eval": n_eval,
        "evaluations_recorded": len(store["X"]),
        "decodes": int(getattr(problem, "decodes_total", n_eval)),
        "repaired_evaluations": int(getattr(problem, "repaired_count", 0)),
        "runtime_s": runtime,
        "eval_time_s": float(getattr(problem, "eval_time_s", float("nan"))),
        "feasible_evaluations": int(np.sum(np.all(np.vstack(store["G"]) <= 0, axis=1))),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "dataset_file": str(ds_path.relative_to(ROOT)),
    }
    run_path.parent.mkdir(parents=True, exist_ok=True)
    run_path.write_text(json.dumps(summary, indent=2))
    return summary, "ran"


def write_summary(out_dir):
    rows = []
    for p in sorted((out_dir / "runs").glob("*.json")):
        rows.append(json.loads(p.read_text()))
    if not rows:
        return
    keys = list(rows[0].keys())
    with open(out_dir / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["SC1-50", "SC1-100", "SC1-250", "SC1-500"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    ap.add_argument("--methods", nargs="+", default=list(METHODS), choices=METHODS)
    ap.add_argument("--out", default="experiments/reduced_sc1")
    args = ap.parse_args()

    out_dir = (ROOT / args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps({
        "datasets": args.datasets, "seeds": args.seeds, "methods": args.methods,
        "budget_rule": "random uses the NSGA-II evaluation count for the same dataset and seed",
        "population_rule": "N<=100: 100/100; N>=250: 50/50 (as phase5_runner.py)",
        "repair_rule": "see optimization/nsga2/feasibility.py",
    }, indent=2))

    for ds in args.datasets:
        for seed in args.seeds:
            # NSGA-II first: its evaluation count defines the budget for random search.
            budget_info = None
            for method in ("nsga2", "nsga2_repair", "random"):
                if method not in args.methods:
                    continue
                if method == "random":
                    nsga_run = json.loads(
                        (out_dir / "runs" / f"{ds}__nsga2__seed{seed}.json").read_text())
                    budget_info = {"n_eval": nsga_run["n_eval"]}
                summary, status = run_one(ds, method, seed, budget_info, out_dir)
                print(f"[{status}] {ds} {method} seed={seed} "
                      f"n_eval={summary['n_eval']} runtime={summary['runtime_s']:.1f}s "
                      f"feasible={summary['feasible_evaluations']}", flush=True)
                write_summary(out_dir)


if __name__ == "__main__":
    main()
