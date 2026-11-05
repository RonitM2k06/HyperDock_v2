"""
Ablation study of the feasibility-aware variant (nsga2_repair).

Each mode changes one component of the variant. Everything else (operators,
population, generations, seeds) matches experiments/matched_budget_runner.py, so
the "full" reference for seeds 1-10 is the existing nsga2_repair runs in
experiments/full_matched (no re-run needed; same code, same seeds).

Modes
  no_orientation    orientation genes ignored (every item uses rotation 1)
  fallback_shuffle  fallback decode uses a random order instead of descending volume
                    (tests whether the gain comes from volume ordering or from a
                    second decode attempt as such)
  no_reach          CASC optimised without the reach term     (L_arm -> inf)
  no_occ            CASC optimised without the occlusion term (lambda_occ = 0)
  no_ext            CASC optimised without the extraction term (C_grasp = kappa = 0)
  supported         decoder requires contact: an item not at z = 0 must have at
                    least 50% of its footprint on faces of placed items at its z

Evaluation: the final feasible non-dominated set of each run (under the mode's own
objective) is re-decoded with the mode's decoder and re-scored with the FULL
objective (default CoM and CASC). The analysis compares these true-objective
fronts with the full variant on the same seeds.

Outputs: <out>/fronts/<dataset>__<mode>__seed<s>.npz  (X, F_true, G)
         <out>/runs/<dataset>__<mode>__seed<s>.json
Runs are independent; --workers runs them in parallel processes. Existing runs
are skipped, so an interrupted batch resumes.

Usage (repository root):
  python experiments/ablation_study.py --datasets SC1-100 SC2-100 --seeds 1 2 3 4 5 6 7 8 9 10 --workers 12
"""
import argparse
import json
import os
import platform
import sys
import time
import zlib
from pathlib import Path
from multiprocessing import Pool

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

MODES = ("no_orientation", "fallback_shuffle", "no_reach", "no_occ", "no_ext", "supported")
DATASET_DIR = ROOT / "datasets" / "hyperdock_bench"


def _volume(d):
    return float(d[0]) * float(d[1]) * float(d[2])


def make_supported_decoder_cls():
    from core.placement.extreme_point import ExtremePointDecoder, can_pack_jit

    class SupportedEPDecoder(ExtremePointDecoder):
        """EP decoder that also requires >= 50% footprint contact at the item's z."""
        MIN_SUPPORT = 0.5

        def _supported(self, ep, dims):
            x, y, z = ep
            w, d, _ = dims
            if z < 1e-5:
                return True
            area = 0.0
            for it in self.packed_items:
                px, py, pz = it["point"]
                pw, pd, ph = it["dimensions"]
                if abs(pz + ph - z) > 1e-4:
                    continue
                ox = min(x + w, px + pw) - max(x, px)
                oy = min(y + d, py + pd) - max(y, py)
                if ox > 0 and oy > 0:
                    area += ox * oy
            return area >= self.MIN_SUPPORT * w * d - 1e-9

        def try_pack_item(self, dimensions, item_data):
            dims = np.array(dimensions, dtype=np.float64)
            for ep in self.extreme_points:
                pt = np.array(ep, dtype=np.float64)
                if can_pack_jit(pt, dims, self.cw, self.cd, self.ch,
                                self.packed_points_arr, self.packed_dims_arr, self.num_packed):
                    if self._supported(ep, dimensions):
                        self.add_item(ep, dimensions, item_data)
                        return True
            return False

    return SupportedEPDecoder


def build_problem(items, container_dims, mode):
    """Variant problem with one component changed. mode=None reproduces CargoProblemRepair."""
    from core.placement.extreme_point import ExtremePointDecoder
    from core.placement.rotation import RotationEngine
    from core.metrics.casc import CASCv2, CASCv2Config
    from optimization.nsga2.problem import CargoProblem

    decoder_cls = make_supported_decoder_cls() if mode == "supported" else ExtremePointDecoder
    casc_cfg = CASCv2Config(grid_resolution=10)
    if mode == "no_reach":
        casc_cfg.L_arm = 1e12
    elif mode == "no_occ":
        casc_cfg.lambda_occ = 0.0
    elif mode == "no_ext":
        casc_cfg.C_grasp = 0.0
        casc_cfg.kappa = 0.0

    class AblationProblem(CargoProblem):
        def __init__(self):
            super().__init__(items=items, container_dims=container_dims)
            self.casc_evaluator = CASCv2(casc_cfg)   # objective used for search
            self.full_casc = CASCv2(CASCv2Config(grid_resolution=10))
            self.decodes_total = 0

        def decode(self, sequence, orientations):
            dec = decoder_cls(self.container_dims)
            uv, uc = 0.0, 0
            for idx in sequence:
                base = self.items[idx]["dimensions_wdh_meters"]
                rots = RotationEngine.all_rotations(base)
                r = 0 if mode == "no_orientation" else max(0, min(int(orientations[idx]) - 1, len(rots) - 1))
                if not dec.try_pack_item(rots[r], self.items[idx]):
                    uv += _volume(base)
                    uc += 1
            return uv, uc, dec.packed_items

        def decode_with_fallback(self, x):
            seq = [int(v) for v in x[: self.n_items]]
            ori = x[self.n_items:].astype(int)
            uv, uc, packed = self.decode(seq, ori)
            dec = 1
            if uc > 0:
                if mode == "fallback_shuffle":
                    rng = np.random.default_rng(zlib.crc32(np.asarray(x, np.int64).tobytes()))
                    seq2 = [int(v) for v in rng.permutation(seq)]
                else:
                    seq2 = sorted(seq, key=lambda i: -_volume(self.items[i]["dimensions_wdh_meters"]))
                uv2, uc2, packed2 = self.decode(seq2, ori)
                dec += 1
                if uv2 < uv:
                    uv, uc, packed = uv2, uc2, packed2
            return uv, uc, packed, dec

        def score(self, packed, casc):
            if len(packed) == 0:
                return 9999.0, 0.0
            return (float(self.com_evaluator.evaluate_layout(packed)["delta_com"]),
                    -float(casc.evaluate_layout(packed)["mean_casc"]))

        def _evaluate(self, x, out, *args, **kwargs):
            uv, uc, packed, dec = self.decode_with_fallback(x)
            self.decodes_total += dec
            out["F"] = list(self.score(packed, self.casc_evaluator))
            out["G"] = [uv, float(uc)]

    return AblationProblem()


def nondominated_mask(F):
    keep = np.ones(len(F), bool)
    for i in range(len(F)):
        if keep[i] and np.any(np.all(F <= F[i], axis=1) & np.any(F < F[i], axis=1)):
            keep[i] = False
    return keep


def run_one(args):
    dataset, mode, seed, out_dir, gens_override = args
    out_dir = Path(out_dir)
    stem = f"{dataset}__{mode}__seed{seed}"
    run_path = out_dir / "runs" / f"{stem}.json"
    front_path = out_dir / "fronts" / f"{stem}.npz"
    if run_path.exists() and front_path.exists():
        return stem, "skipped"

    from pymoo.algorithms.moo.nsga2 import NSGA2
    from pymoo.optimize import minimize
    from optimization.nsga2.operators import CargoCrossover, CargoMutation, CargoSampling

    data = json.loads((DATASET_DIR / f"HB-V1-{dataset}.json").read_text())
    items, cdims = data["items"], data["containers"][0]["dimensions_wdh_meters"]
    n = len(items)
    pop, gens = (50, 50) if n >= 250 else (100, 100)
    if gens_override:  # smoke tests only; results with an override are not comparable
        pop, gens = 20, gens_override
    np.random.seed(seed)
    problem = build_problem(items, cdims, mode)

    archive_X, archive_F, archive_G = [], [], []
    orig = problem._evaluate

    def rec(x, out, *a, **k):
        orig(x, out, *a, **k)
        archive_X.append(np.asarray(x, float).copy())
        archive_F.append(np.asarray(out["F"], float).copy())
        archive_G.append(np.asarray(out["G"], float).copy())
    problem._evaluate = rec

    t0 = time.perf_counter()
    algorithm = NSGA2(pop_size=pop, sampling=CargoSampling(n_items=n),
                      crossover=CargoCrossover(n_items=n, prob=0.9),
                      mutation=CargoMutation(n_items=n, prob=1.0 / n),
                      eliminate_duplicates=True)
    res = minimize(problem, algorithm, ("n_gen", gens), seed=seed, verbose=False)
    runtime = time.perf_counter() - t0

    X, F, G = np.vstack(archive_X), np.vstack(archive_F), np.vstack(archive_G)
    feas = np.all(G <= 0, axis=1)
    Xf, Ff = X[feas], F[feas]
    if len(Ff):
        Ff, idx = np.unique(Ff, axis=0, return_index=True)
        Xf = Xf[idx]
        k = nondominated_mask(Ff)
        Xf = Xf[k]
    # Re-score the mode's front with the full objective.
    F_true, G_true = [], []
    for x in Xf:
        uv, uc, packed, _ = problem.decode_with_fallback(x)
        F_true.append(problem.score(packed, problem.full_casc))
        G_true.append((uv, float(uc)))
    front_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(front_path, X=np.asarray(Xf, float).reshape(len(Xf), 2 * n),
                        F_true=np.asarray(F_true).reshape(-1, 2), G=np.asarray(G_true).reshape(-1, 2))
    meta = {"dataset": dataset, "mode": mode, "seed": seed, "n_items": n,
            "n_eval": int(res.algorithm.evaluator.n_eval), "decodes": int(problem.decodes_total),
            "runtime_s": runtime, "feasible_evaluations": int(feas.sum()),
            "front_size": int(len(Xf)), "python": platform.python_version(),
            "platform": platform.platform(), "cpu_count": os.cpu_count()}
    run_path.parent.mkdir(parents=True, exist_ok=True)
    run_path.write_text(json.dumps(meta, indent=2))
    return stem, f"ran {runtime:.0f}s feasible_evals={meta['feasible_evaluations']} front={meta['front_size']}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["SC1-100", "SC2-100"])
    ap.add_argument("--seeds", nargs="+", type=int, default=list(range(1, 11)))
    ap.add_argument("--modes", nargs="+", default=list(MODES), choices=MODES)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--out", default="experiments/ablation")
    ap.add_argument("--smoke-gens", type=int, default=0, help="smoke test only")
    a = ap.parse_args()
    out = (ROOT / a.out).resolve()
    jobs = [(d, m, s, str(out), a.smoke_gens) for d in a.datasets for m in a.modes for s in a.seeds]
    print(f"{len(jobs)} runs, {a.workers} workers", flush=True)
    if a.workers > 1:
        with Pool(a.workers) as pool:
            for stem, status in pool.imap_unordered(run_one, jobs):
                print(f"[{status}] {stem}", flush=True)
    else:
        for j in jobs:
            stem, status = run_one(j)
            print(f"[{status}] {stem}", flush=True)


if __name__ == "__main__":
    main()
