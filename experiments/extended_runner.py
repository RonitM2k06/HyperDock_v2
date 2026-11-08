"""
Extended experiments for manuscript v7, in the matched-budget output format
(runs/*.json, raw/*.npz), run in parallel.

Methods
  nsga2               plain NSGA-II                         (as matched_budget_runner)
  nsga2_repair        volume-ordered Baldwinian repair      (as matched_budget_runner)
  random              random search, budget = nsga2 n_eval  (needs the nsga2 run first)
  nsga2_seeded        plain NSGA-II whose initial population contains one BFD-V
                      chromosome (decreasing-volume order with the orientations BFD-V
                      chose); the rest is random. Injects the same constructive prior
                      as the repair, through initialisation instead of decoding.
  nsga2_repair_large  repair that moves only items with volume >= the median to the
                      front (decreasing volume), keeping the chromosome's order for the
                      smaller items. Tests whether the search can keep control of the
                      small items (and so accessibility) while the large items secure
                      feasibility.
  bfdv                deterministic BFD-V (all six rotations per EP), one run, seed 0.

Decoders
  standard            EP decoder of the paper
  contact             EP decoder that also requires >= 50% footprint contact with items
                      directly nearer the access face (or z = 0)

Usage (repository root)
  # Stage 1: new methods on the full-run protocol (standard decoder)
  python experiments/extended_runner.py --stage new_methods --workers 14
  # Stage 1: contact-decoder study
  python experiments/extended_runner.py --stage contact --workers 14
  # Stage 2 (optional): seeds 11-30 for N >= 250
  python experiments/extended_runner.py --stage more_seeds --workers 14
Finished runs are skipped, so any stage can be interrupted and restarted.
"""
import argparse
import json
import os
import platform
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

DATASET_DIR = ROOT / "datasets" / "hyperdock_bench"
SC12 = [f"{s}-{n}" for s in ("SC1", "SC2") for n in (50, 100, 250, 500)]


def _vol(d):
    return float(d[0]) * float(d[1]) * float(d[2])


BR_DIR = ROOT / "datasets" / "br"
BR = ["BR1-1", "BR1-2", "BR4-1", "BR4-2", "BR7-1", "BR7-2",
      "BR8-1", "BR8-2", "BR9-1", "BR9-2", "BR10-1", "BR10-2"]


def _dataset(ds):
    path = BR_DIR / f"{ds}.json" if ds.startswith("BR") else DATASET_DIR / f"HB-V1-{ds}.json"
    return json.loads(path.read_text())


def load(ds):
    d = _dataset(ds)
    return d["items"], d["containers"][0]["dimensions_wdh_meters"]


def com_target(ds):
    return _dataset(ds).get("com_target")


def allowed_rotations(item):
    """0-based RotationEngine indices this item may use (all six unless restricted)."""
    return item.get("allowed_rotations", list(range(6)))


def rotation_for_gene(item, gene):
    """Map an orientation gene in 1..6 onto the item's allowed rotations.
    With all six allowed this is the identity (gene g -> rotation g)."""
    from core.placement.rotation import RotationEngine
    allowed = allowed_rotations(item)
    rots = RotationEngine.all_rotations(item["dimensions_wdh_meters"])
    g = max(1, min(int(gene), 6))
    return rots[allowed[(g - 1) % len(allowed)]]


def decoder_class(name):
    from core.placement.extreme_point import ExtremePointDecoder
    if name == "standard":
        return ExtremePointDecoder
    from experiments.ablation_study import make_supported_decoder_cls
    return make_supported_decoder_cls()


def build_problem(items, cdims, decoder, repair, target=None):
    """repair in {None, 'volume', 'large'}. decoder in {'standard','contact'}.
    With decoder='standard', repair=None equals CargoProblem and repair='volume'
    equals CargoProblemRepair (tested)."""
    from core.placement.rotation import RotationEngine
    from optimization.nsga2.problem import CargoProblem
    dec_cls = decoder_class(decoder)
    vols = np.array([_vol(it["dimensions_wdh_meters"]) for it in items])
    median = float(np.median(vols))

    class P(CargoProblem):
        def __init__(self):
            super().__init__(items=items, container_dims=cdims)
            if target is not None:
                from core.metrics.com import CoMModel, CoMConfig
                t = np.array(target, dtype=float)
                self.com_evaluator = CoMModel(CoMConfig(target_com=t, dry_mass=5000.0, dry_com=t))
            self.decodes_total = 0
            self.repaired_count = 0
            self.repair_active = True   # FAPS switches this on demand

        def decode(self, seq, ori):
            d = dec_cls(self.container_dims)
            uv, uc = 0.0, 0
            for idx in seq:
                base = self.items[idx]["dimensions_wdh_meters"]
                if not d.try_pack_item(rotation_for_gene(self.items[idx], ori[idx]), self.items[idx]):
                    uv += _vol(base)
                    uc += 1
            return uv, uc, d.packed_items

        def _evaluate(self, x, out, *a, **k):
            seq = [int(v) for v in x[: self.n_items]]
            ori = np.asarray(x[self.n_items:]).astype(int)
            uv, uc, packed = self.decode(seq, ori)
            self.decodes_total += 1
            if uc > 0 and repair is not None and self.repair_active:
                if repair == "volume":
                    seq2 = sorted(seq, key=lambda i: -vols[i])
                else:  # 'large': large items first by volume, small items keep GA order
                    big = sorted([i for i in seq if vols[i] >= median], key=lambda i: -vols[i])
                    seq2 = big + [i for i in seq if vols[i] < median]
                uv2, uc2, p2 = self.decode(seq2, ori)
                self.decodes_total += 1
                if uv2 < uv:
                    uv, uc, packed = uv2, uc2, p2
                    self.repaired_count += 1
            if len(packed) == 0:
                f = [9999.0, 0.0]
            else:
                f = [float(self.com_evaluator.evaluate_layout(packed)["delta_com"]),
                     -float(self.casc_evaluator.evaluate_layout(packed)["mean_casc"])]
            out["F"] = f
            out["G"] = [uv, float(uc)]

    return P()


PORTFOLIO = {
    # name: sort key (sorted descending); the first placed items sit nearest the access face
    "volume": lambda it: _vol(it["dimensions_wdh_meters"]),
    "base_area": lambda it: max(a * b for a, b in [(it["dimensions_wdh_meters"][0], it["dimensions_wdh_meters"][1]),
                                                   (it["dimensions_wdh_meters"][0], it["dimensions_wdh_meters"][2]),
                                                   (it["dimensions_wdh_meters"][1], it["dimensions_wdh_meters"][2])]),
    "longest_side": lambda it: max(it["dimensions_wdh_meters"]),
    "mass": lambda it: float(it.get("mass_kg", 0.0)),
}


def bfdv_chromosome(items, cdims, decoder):
    """Deterministic BFD-V: decreasing volume, first EP then first rotation that fits."""
    return constructive_chromosome(items, cdims, decoder, "volume")


def constructive_chromosome(items, cdims, decoder, key):
    """Best-fit-decreasing by PORTFOLIO[key]: items sorted by decreasing key (stable),
    each placed at the first extreme point and first permitted rotation that fits.
    Returns (sequence, orientation genes 1..6 that reproduce the layout, packed list)."""
    from core.placement.rotation import RotationEngine
    from core.placement.extreme_point import can_pack_jit
    d = decoder_class(decoder)(cdims)
    f = PORTFOLIO[key]
    seq = sorted(range(len(items)), key=lambda i: -f(items[i]))
    ori = np.ones(len(items), dtype=int)
    for i in seq:
        all_rots = RotationEngine.all_rotations(items[i]["dimensions_wdh_meters"])
        rots = [all_rots[k] for k in allowed_rotations(items[i])]
        done = False
        for ep in list(d.extreme_points):
            pt = np.array(ep, dtype=np.float64)
            for r, dims in enumerate(rots):
                dd = np.array(dims, dtype=np.float64)
                ok = can_pack_jit(pt, dd, d.cw, d.cd, d.ch, d.packed_points_arr,
                                  d.packed_dims_arr, d.num_packed)
                if ok and hasattr(d, "_supported"):
                    ok = d._supported(ep, dims)
                if ok:
                    d.add_item(ep, dims, items[i])
                    ori[i] = r + 1
                    done = True
                    break
            if done:
                break
    return np.array(seq), ori, d.packed_items


def run_job(job):
    ds, method, seed, decoder, out_dir = job
    out_dir = Path(out_dir)
    stem = f"{ds}__{method}__seed{seed}"
    raw_p, run_p = out_dir / "raw" / f"{stem}.npz", out_dir / "runs" / f"{stem}.json"
    if raw_p.exists() and run_p.exists():
        return stem, "skipped"
    from pymoo.algorithms.moo.nsga2 import NSGA2
    from pymoo.optimize import minimize
    from pymoo.core.population import Population
    from optimization.nsga2.operators import CargoCrossover, CargoMutation, CargoSampling

    items, cdims = load(ds)
    n = len(items)
    pop, gens = (50, 50) if n >= 250 else (100, 100)
    if os.environ.get("HD_SMOKE_GENS"):  # smoke tests only; never used for reported runs
        pop, gens = 20, int(os.environ["HD_SMOKE_GENS"])
    np.random.seed(seed)
    t0 = time.perf_counter()

    if method == "bfdv":
        seq, ori, packed = bfdv_chromosome(items, cdims, decoder)
        p = build_problem(items, cdims, decoder, None, com_target(ds))
        x = np.concatenate([seq, ori]).astype(float)
        uv = sum(_vol(it["dimensions_wdh_meters"]) for it in items) - sum(_vol(q["dimensions"]) for q in packed)
        uc = n - len(packed)
        f = [9999.0, 0.0] if not packed else [
            float(p.com_evaluator.evaluate_layout(packed)["delta_com"]),
            -float(p.casc_evaluator.evaluate_layout(packed)["mean_casc"])]
        X, F, G = x[None, :], np.array([f]), np.array([[max(0.0, uv), float(uc)]])
        n_eval, decodes = 1, 1
    else:
        base = method[4:] if method.startswith("sms_") else method
        base = {"": "nsga2", "repair": "nsga2_repair", "seeded": "nsga2_seeded", "faps": "faps"}.get(base, base)
        portfolio, faps_mode = None, None
        if base == "faps":
            # Feasibility-adaptive portfolio seeding (FAPS): one constructive seed per
            # portfolio order, and the volume repair held in reserve. The repair is switched
            # on only if no feasible packing has been evaluated after 10% of the generations.
            portfolio = [constructive_chromosome(items, cdims, decoder, k) for k in PORTFOLIO]
            faps_mode = {"feasible_seeds": int(sum(len(pk) == n for _, _, pk in portfolio)),
                         "repair_activated_at_gen": None}
        repair = {"nsga2": None, "nsga2_seeded": None, "random": None, "faps": "volume",
                  "nsga2_repair": "volume", "nsga2_repair_large": "large"}[base]
        prob = build_problem(items, cdims, decoder, repair, com_target(ds))
        if base == "faps":
            prob.repair_active = False
        store = {"X": [], "F": [], "G": []}
        orig = prob._evaluate

        def rec(x, out, *a, **k):
            orig(x, out, *a, **k)
            store["X"].append(np.asarray(x, float).copy())
            store["F"].append(np.asarray(out["F"], float).copy())
            store["G"].append(np.asarray(out["G"], float).copy())
        prob._evaluate = rec

        if method == "random":
            ref = json.loads((out_dir / "runs" / f"{ds}__nsga2__seed{seed}.json").read_text())
            n_eval = int(ref["n_eval"])
            for x in CargoSampling(n_items=n)._do(prob, n_eval):
                prob._evaluate(x, {})
        else:
            sampling = CargoSampling(n_items=n)
            seeds = []
            if base == "nsga2_seeded":
                seq, ori, _ = bfdv_chromosome(items, cdims, decoder)
                seeds = [(seq, ori)]
            elif base == "faps":
                seen = set()
                for seq, ori, _ in portfolio:
                    key = (tuple(seq), tuple(ori))
                    if key not in seen:
                        seen.add(key)
                        seeds.append((seq, ori))
            if seeds:
                X0 = CargoSampling(n_items=n)._do(prob, pop)
                for j, (seq, ori) in enumerate(seeds):
                    X0[j, :n], X0[j, n:] = seq, ori
                sampling = Population.new("X", X0)
            ops = dict(pop_size=pop, sampling=sampling,
                       crossover=CargoCrossover(n_items=n, prob=0.9),
                       mutation=CargoMutation(n_items=n, prob=1.0 / n),
                       eliminate_duplicates=True)
            if method.startswith("sms_"):
                from pymoo.algorithms.moo.sms import SMSEMOA
                alg = SMSEMOA(**ops)
            else:
                alg = NSGA2(**ops)
            callback = None
            if base == "faps":
                check_gen = max(1, int(round(0.1 * gens)))

                def callback(algorithm):
                    if (not prob.repair_active and algorithm.n_gen >= check_gen
                            and not any(np.all(g <= 0) for g in store["G"])):
                        prob.repair_active = True
                        faps_mode["repair_activated_at_gen"] = int(algorithm.n_gen)
            kw = {"callback": callback} if callback is not None else {}
            res = minimize(prob, alg, ("n_gen", gens), seed=seed, verbose=False, **kw)
            n_eval = int(res.algorithm.evaluator.n_eval)
        X, F, G = np.vstack(store["X"]), np.vstack(store["F"]), np.vstack(store["G"])
        decodes = int(prob.decodes_total)

    runtime = time.perf_counter() - t0
    raw_p.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(raw_p, X=X, F=F, G=G, n_items=np.array(n))
    meta = {"dataset": ds, "method": method, "seed": seed, "n_items": n, "decoder": decoder,
            "n_eval": n_eval, "decodes": decodes + (len(PORTFOLIO) if method.endswith("faps") else 0),
            "runtime_s": runtime, "faps_mode": locals().get("faps_mode"),
            "feasible_evaluations": int(np.all(G <= 0, axis=1).sum()),
            "python": platform.python_version(), "platform": platform.platform(),
            "cpu_count": os.cpu_count()}
    run_p.parent.mkdir(parents=True, exist_ok=True)
    run_p.write_text(json.dumps(meta, indent=2))
    return stem, f"ran {runtime:.0f}s feasible_evals={meta['feasible_evaluations']}"


def seeds_for(ds, extended=False):
    n = int(ds.split("-")[1])
    return list(range(1, 31)) if n <= 100 else (list(range(11, 31)) if extended else list(range(1, 11)))


def jobs_for(stage):
    if stage == "new_methods":
        out = str(ROOT / "experiments" / "full_matched")
        phase1 = [(d, m, s, "standard", out) for d in SC12 for m in ("nsga2_seeded", "nsga2_repair_large")
                  for s in seeds_for(d)]
        return [phase1]
    if stage == "contact":
        out = str(ROOT / "experiments" / "contact")
        dss = ["SC1-50", "SC2-50", "SC1-100", "SC2-100"]
        phase1 = [(d, m, s, "contact", out) for d in dss
                  for m in ("nsga2", "nsga2_repair", "nsga2_seeded", "nsga2_repair_large")
                  for s in range(1, 11)] + [(d, "bfdv", 0, "contact", out) for d in dss]
        phase2 = [(d, "random", s, "contact", out) for d in dss for s in range(1, 11)]
        return [phase1, phase2]
    if stage == "br":
        out = str(ROOT / "experiments" / "br")
        phase1 = [(d, m, s, "standard", out) for d in BR
                  for m in ("nsga2", "nsga2_repair", "nsga2_seeded", "nsga2_repair_large")
                  for s in range(1, 11)] + [(d, "bfdv", 0, "standard", out) for d in BR]
        phase2 = [(d, "random", s, "standard", out) for d in BR for s in range(1, 11)]
        return [phase1, phase2]
    if stage == "faps":
        out_s = str(ROOT / "experiments" / "full_matched")
        out_b = str(ROOT / "experiments" / "br")
        p1 = [(d, "faps", s, "standard", out_s) for d in SC12 for s in range(1, 31)]
        p1 += [(d, "faps", s, "standard", out_b) for d in BR for s in range(1, 11)]
        return [p1]
    if stage == "sms":
        out = str(ROOT / "experiments" / "sms")
        dss = ["SC1-50", "SC2-50", "SC1-100", "SC1-250", "SC2-250", "BR1-1", "BR4-2", "BR7-2"]
        p1 = [(d, m, s, "standard", out) for d in dss
              for m in ("sms_nsga2", "sms_repair", "sms_seeded", "sms_faps") for s in range(1, 11)]
        return [p1]  # random search does not depend on the optimiser; see the main study
    if stage == "more_seeds":
        out = str(ROOT / "experiments" / "full_matched")
        dss = [d for d in SC12 if int(d.split("-")[1]) >= 250]
        phase1 = [(d, m, s, "standard", out) for d in dss
                  for m in ("nsga2", "nsga2_repair", "nsga2_seeded", "nsga2_repair_large")
                  for s in seeds_for(d, extended=True)]
        phase2 = [(d, "random", s, "standard", out) for d in dss for s in seeds_for(d, extended=True)]
        return [phase1, phase2]
    raise ValueError(stage)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["new_methods", "contact", "more_seeds", "br", "faps", "sms"])
    ap.add_argument("--workers", type=int, default=12)
    a = ap.parse_args()
    for phase in jobs_for(a.stage):
        # longest jobs first, for better load balance
        phase.sort(key=lambda j: -len(load(j[0])[0]))
        print(f"{len(phase)} runs, {a.workers} workers", flush=True)
        with Pool(a.workers) as pool:
            for stem, status in pool.imap_unordered(run_job, phase):
                print(f"[{status}] {stem}", flush=True)


if __name__ == "__main__":
    main()
