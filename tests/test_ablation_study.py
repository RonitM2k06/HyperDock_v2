"""The ablation problem with no change must reproduce the variant exactly."""
import json
from pathlib import Path
import numpy as np
from experiments.ablation_study import build_problem
from optimization.nsga2.feasibility import CargoProblemRepair

ROOT = Path(__file__).resolve().parent.parent


def test_unmodified_ablation_problem_matches_variant():
    d = json.loads((ROOT / "datasets/hyperdock_bench/HB-V1-SC1-50.json").read_text())
    items, c = d["items"], d["containers"][0]["dimensions_wdh_meters"]
    a, b = build_problem(items, c, None), CargoProblemRepair(items, c)
    rng = np.random.default_rng(0)
    for _ in range(10):
        x = np.concatenate([rng.permutation(50), rng.integers(1, 7, 50)]).astype(float)
        oa, ob = {}, {}
        a._evaluate(x, oa); b._evaluate(x, ob)
        assert np.allclose(oa["F"], ob["F"]) and np.allclose(oa["G"], ob["G"])


def test_supported_decoder_never_floats():
    d = json.loads((ROOT / "datasets/hyperdock_bench/HB-V1-SC1-50.json").read_text())
    items, c = d["items"], d["containers"][0]["dimensions_wdh_meters"]
    p = build_problem(items, c, "supported")
    x = np.concatenate([np.arange(50), np.ones(50)]).astype(float)
    _, _, packed, _ = p.decode_with_fallback(x)
    for it in packed:
        z = it["point"][2]
        if z > 1e-5:
            assert any(abs(o["point"][2] + o["dimensions"][2] - z) < 1e-4 for o in packed)
