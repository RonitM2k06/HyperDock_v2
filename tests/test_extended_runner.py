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


def test_br_orientation_flags_respected():
    """Every placed BR box stands only on a dimension its BR flag allows to be vertical (y)."""
    from experiments.extended_runner import rotation_for_gene, allowed_rotations
    from experiments.br_instances import PERMS
    items, c = load("BR4-1")
    rng = np.random.default_rng(0)
    for it in items:
        allowed_vertical = {it["dimensions_wdh_meters"][PERMS[r][1]] for r in allowed_rotations(it)}
        for g in range(1, 7):
            assert rotation_for_gene(it, g)[1] in allowed_vertical
    p = build_problem(items, c, "standard", "volume")
    x = np.concatenate([rng.permutation(len(items)), rng.integers(1, 7, len(items))]).astype(float)
    _, _, packed = p.decode([int(v) for v in x[:len(items)]], x[len(items):].astype(int))
    assert len(packed) > 0


def test_br_com_target_is_container_centre():
    from experiments.extended_runner import com_target
    items, c = load("BR1-1")
    assert np.allclose(com_target("BR1-1"), np.array(c) / 2)


def test_bfdv_chromosome_reproduces_on_br():
    from experiments.extended_runner import com_target
    for ds in ["BR1-1", "BR7-1"]:
        items, c = load(ds)
        seq, ori, packed = bfdv_chromosome(items, c, "standard")
        p = build_problem(items, c, "standard", None, com_target(ds))
        out = {}
        p._evaluate(np.concatenate([seq, ori]).astype(float), out)
        assert out["G"][1] == len(items) - len(packed)


def test_portfolio_chromosomes_reproduce_their_layouts():
    from experiments.extended_runner import constructive_chromosome, PORTFOLIO
    items, c = load("SC1-250")
    p = build_problem(items, c, "standard", None)
    for k in PORTFOLIO:
        seq, ori, packed = constructive_chromosome(items, c, "standard", k)
        out = {}
        p._evaluate(np.concatenate([seq, ori]).astype(float), out)
        assert out["G"][1] == len(items) - len(packed)


def test_faps_repair_is_inactive_until_switched_on():
    items, c = load("SC2-50")
    p = build_problem(items, c, "standard", "volume")
    q = build_problem(items, c, "standard", None)
    p.repair_active = False
    for x in _chromosomes(50, 5, 9):
        a, b = {}, {}
        p._evaluate(x, a); q._evaluate(x, b)
        assert np.allclose(a["F"], b["F"]) and np.allclose(a["G"], b["G"])
