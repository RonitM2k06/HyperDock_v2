"""Tests for the feasibility-aware variant (optimization/nsga2/feasibility.py)."""
import numpy as np

from optimization.nsga2.problem import CargoProblem
from optimization.nsga2.feasibility import (
    CargoProblemRepair, decode_layout, descending_volume_order)


def _item(i, dims, mass=10.0):
    return {"item_id": str(i), "dimensions_wdh_meters": dims, "mass_kg": mass}


def _rng_items(n, seed=0):
    rng = np.random.default_rng(seed)
    items = []
    for i in range(n):
        d = rng.uniform(0.15, 0.4, size=3).round(3).tolist()
        items.append(_item(i, d, mass=float(rng.uniform(1, 20))))
    return items


def test_decode_matches_base_problem_constraints():
    """decode_layout must reproduce CargoProblem's G exactly for the same chromosome."""
    items = _rng_items(12, seed=3)
    container = [1.0, 1.0, 1.0]
    base = CargoProblem(items=items, container_dims=container)
    rng = np.random.default_rng(7)
    for _ in range(15):
        seq = rng.permutation(12)
        ori = rng.integers(1, 7, size=12)
        x = np.concatenate([seq, ori]).astype(float)
        out = {}
        base._evaluate(x, out)
        uv, uc, _ = decode_layout(items, container, seq, ori)
        assert np.isclose(out["G"][0], uv)
        assert out["G"][1] == uc


def test_repair_never_modifies_genotype():
    items = _rng_items(10, seed=11)
    prob = CargoProblemRepair(items=items, container_dims=[0.9, 0.9, 0.9])
    rng = np.random.default_rng(5)
    for _ in range(10):
        x = np.concatenate([rng.permutation(10), rng.integers(1, 7, size=10)]).astype(float)
        before = x.copy()
        out = {}
        prob._evaluate(x, out)
        assert np.array_equal(x, before)


def test_repair_never_increases_unplaced_volume():
    items = _rng_items(14, seed=21)
    container = [0.9, 0.9, 0.9]
    base = CargoProblem(items=items, container_dims=container)
    rep = CargoProblemRepair(items=items, container_dims=container)
    rng = np.random.default_rng(9)
    for _ in range(20):
        x = np.concatenate([rng.permutation(14), rng.integers(1, 7, size=14)]).astype(float)
        ob, orp = {}, {}
        base._evaluate(x.copy(), ob)
        rep._evaluate(x.copy(), orp)
        assert orp["G"][0] <= ob["G"][0] + 1e-12


def test_constructed_case_repair_lowers_unplaced_volume():
    """
    Container 1x1x1. Sequence [S, B]: the item S (0.2 x 0.2 x 0.5) sits in the corner
    and leaves no room for B (1 x 1 x 0.6): 0.5 + 0.6 > 1, so the base decode leaves
    B unplaced (volume 0.6).
    Descending-volume order [B, S] places B first; S then cannot fit on top
    (0.6 + 0.5 > 1), so S is unplaced (volume 0.02). That is less unplaced volume,
    so the repair keeps it. Both decodes are infeasible: the repair lowers the
    constraint violation without reaching feasibility, and this test asserts exactly that.
    """
    items = [
        _item(0, [0.2, 0.2, 0.5], mass=5.0),   # S
        _item(1, [1.0, 1.0, 0.6], mass=50.0),  # B
    ]
    container = [1.0, 1.0, 1.0]
    x = np.array([0, 1, 1, 1], dtype=float)    # sequence [S, B], orientations [1, 1]

    base = CargoProblem(items=items, container_dims=container)
    ob = {}
    base._evaluate(x.copy(), ob)
    assert ob["G"][1] == 1 and np.isclose(ob["G"][0], 0.6)

    rep = CargoProblemRepair(items=items, container_dims=container)
    orp = {}
    rep._evaluate(x.copy(), orp)
    assert rep.repaired_count == 1
    assert rep.decodes_total == 2
    assert np.isclose(orp["G"][0], 0.2 * 0.2 * 0.5) and orp["G"][1] == 1
    assert orp["G"][0] < ob["G"][0]


def test_feasible_first_pass_uses_one_decode():
    items = [_item(0, [0.3, 0.3, 0.3]), _item(1, [0.3, 0.3, 0.3])]
    rep = CargoProblemRepair(items=items, container_dims=[1.0, 1.0, 1.0])
    out = {}
    rep._evaluate(np.array([0, 1, 1, 1], dtype=float), out)
    assert out["G"] == [0.0, 0.0]
    assert rep.decodes_total == 1
    assert rep.repaired_count == 0


def test_nothing_packable_is_handled():
    """An item larger than the container: no crash, constraint violated, sentinel objective."""
    items = [_item(0, [2.0, 2.0, 2.0])]
    rep = CargoProblemRepair(items=items, container_dims=[1.0, 1.0, 1.0])
    out = {}
    rep._evaluate(np.array([0, 1], dtype=float), out)
    assert out["G"][1] == 1
    assert out["G"][0] > 0
    assert out["F"] == [9999.0, 0.0]


def test_descending_volume_order_is_stable_permutation():
    items = [_item(0, [1, 1, 1]), _item(1, [2, 2, 2]), _item(2, [1, 1, 1])]
    order = descending_volume_order(items, [0, 1, 2])
    assert order == [1, 0, 2]
    assert sorted(order) == [0, 1, 2]
