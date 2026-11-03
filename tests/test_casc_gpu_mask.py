"""
Regression tests for the padding bug in the batched GPU scorer.

Before the fix, CargoProblemGPU padded unplaced items with zero rows and scored
the padding as real items at the origin. Any layout with fewer packed items
than N was then scored wrongly (CASC error up to about 0.1 on SC1-50).
These tests compare the GPU path with the CPU reference on partially packed
layouts, which is exactly the case the bug affected.

They run on whatever device torch selects (CPU in CI, CUDA on a GPU machine).
"""
import numpy as np
import torch

from core.metrics.casc import CASCv2, CASCv2Config
from core.metrics.casc_gpu import CASCv2GPU
from optimization.nsga2.feasibility import decode_layout
from optimization.nsga2.problem import CargoProblem
from optimization.nsga2.problem_gpu import CargoProblemGPU

TOL = 1e-4


def _items(n, seed):
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        d = rng.uniform(0.2, 0.5, size=3).round(3).tolist()
        out.append({"item_id": str(i), "dimensions_wdh_meters": d,
                    "mass_kg": float(rng.uniform(1, 20)), "category": "hardware"})
    return out


def _partial_layouts(items, container, count, seed):
    """Chromosome decodes that leave at least one item unplaced but pack at least one."""
    n = len(items)
    rng = np.random.default_rng(seed)
    found = []
    while len(found) < count:
        seq = rng.permutation(n)
        ori = rng.integers(1, 7, size=n)
        _, uc, packed = decode_layout(items, container, seq, ori)
        if 0 < len(packed) < n:
            found.append(packed)
    return found


def test_gpu_scorer_matches_cpu_on_partially_packed_layouts():
    items = _items(12, seed=3)
    container = [0.9, 0.9, 0.9]
    cpu = CASCv2(CASCv2Config(grid_resolution=10))
    layouts = _partial_layouts(items, container, count=8, seed=5)

    ref = np.array([cpu.evaluate_layout(p)["mean_casc"] for p in layouts])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu = CASCv2GPU(CASCv2Config(grid_resolution=10), device=device)
    P, D, M, C, V = _padded(layouts, len(items), device)
    got = gpu.evaluate_population(P, D, M, C, V).cpu().numpy()

    assert np.max(np.abs(ref - got)) < TOL


def test_padding_without_mask_is_detected():
    """Documents the bug: without the mask, padding rows change the score."""
    items = _items(12, seed=3)
    container = [0.9, 0.9, 0.9]
    cpu = CASCv2(CASCv2Config(grid_resolution=10))
    layouts = _partial_layouts(items, container, count=4, seed=5)
    ref = np.array([cpu.evaluate_layout(p)["mean_casc"] for p in layouts])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu = CASCv2GPU(CASCv2Config(grid_resolution=10), device=device)
    P, D, M, C, _ = _padded(layouts, len(items), device)
    unmasked = gpu.evaluate_population(P, D, M, C).cpu().numpy()

    assert np.max(np.abs(ref - unmasked)) > 1e-3  # the bug is real and the mask is needed


def test_problem_gpu_matches_problem_cpu_on_infeasible_chromosomes():
    items = _items(12, seed=7)
    container = [0.9, 0.9, 0.9]
    base = CargoProblem(items=items, container_dims=container)
    gpu_problem = CargoProblemGPU(items=items, container_dims=container, n_workers=0)
    rng = np.random.default_rng(11)
    X = np.array([np.concatenate([rng.permutation(12), rng.integers(1, 7, size=12)])
                  for _ in range(20)], dtype=float)

    F_cpu, F_gpu, G_cpu, G_gpu = [], [], [], []
    for x in X:  # CargoProblem._evaluate is element-wise
        o_cpu, o_gpu = {}, {}
        base._evaluate(x, o_cpu)
        gpu_problem._evaluate(x[None, :], o_gpu)  # the GPU path is batched; use a batch of one
        F_cpu.append(o_cpu["F"]); G_cpu.append(o_cpu["G"])
        F_gpu.append(np.asarray(o_gpu["F"]).reshape(-1)); G_gpu.append(np.asarray(o_gpu["G"]).reshape(-1))
    F_cpu, F_gpu = np.asarray(F_cpu, float), np.asarray(F_gpu, float)
    G_cpu, G_gpu = np.asarray(G_cpu, float), np.asarray(G_gpu, float)
    assert np.allclose(G_cpu, G_gpu)                       # constraints are identical
    packed_any = F_cpu[:, 0] < 9000                        # exclude the empty-packing sentinel
    assert packed_any.any()
    assert np.max(np.abs(F_cpu[packed_any] - F_gpu[packed_any])) < TOL


def _padded(layouts, n, device):
    P = len(layouts)
    pts = np.zeros((P, n, 3), np.float32)
    dims = np.zeros((P, n, 3), np.float32)
    masses = np.zeros((P, n), np.float32)
    cats = np.zeros((P, n), np.float32)
    valid = np.zeros((P, n), bool)
    for i, packed in enumerate(layouts):
        for j, it in enumerate(packed):
            pts[i, j] = it["point"]
            dims[i, j] = it["dimensions"]
            masses[i, j] = it["data"].get("mass_kg", 0.0)
            cats[i, j] = 1.0
            valid[i, j] = True
    t = lambda a: torch.from_numpy(a).to(device)  # noqa: E731
    return t(pts), t(dims), t(masses), t(cats), t(valid)
