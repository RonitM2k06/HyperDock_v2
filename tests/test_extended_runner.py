"""The generic problem must reproduce the published methods; the BFD-V chromosome must reproduce BFD-V."""
import numpy as np
from experiments.extended_runner import build_problem, bfdv_chromosome, load
from optimization.nsga2.problem import CargoProblem
from optimization.nsga2.feasibility import CargoProblemRepair
from core.placement.baselines import BestFitDecreasingVolumePacker


def _chromosomes(n, k=8, seed=0):
    rng = np.random.default_rng(seed)
    return [np.concatenate([rng.permutation(n), rng.integers(1, 7, n)]).astype(float) for _ in range(k)]


def test_generic_problem_matches_published_methods():
    items, c = load("SC1-50")
    for repair, ref in [(None, CargoProblem(items, c)), ("volume", CargoProblemRepair(items, c))]:
        p = build_problem(items, c, "standard", repair)
        for x in _chromosomes(50):
            a, b = {}, {}
            p._evaluate(x, a); ref._evaluate(x, b)
            assert np.allclose(a["F"], b["F"]) and np.allclose(a["G"], b["G"])


def test_bfdv_chromosome_reproduces_bfdv():
    for ds in ["SC1-100", "SC2-250"]:
        items, c = load(ds)
        pk = BestFitDecreasingVolumePacker(c); pk.pack(items)
        seq, ori, packed = bfdv_chromosome(items, c, "standard")
        assert len(packed) == len(pk.decoder.packed_items) == len(items)
        p = build_problem(items, c, "standard", None)
        out = {}
        p._evaluate(np.concatenate([seq, ori]).astype(float), out)
        assert out["G"][1] == 0  # the seeded chromosome decodes to a feasible packing


def test_large_repair_never_worse_than_no_repair():
    items, c = load("SC2-50")
    a, b = build_problem(items, c, "standard", "large"), build_problem(items, c, "standard", None)
    for x in _chromosomes(50, 6, 3):
        oa, ob = {}, {}
        a._evaluate(x, oa); b._evaluate(x, ob)
        assert oa["G"][0] <= ob["G"][0] + 1e-12
