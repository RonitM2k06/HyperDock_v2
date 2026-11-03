"""
Scoring-stage CPU-versus-GPU benchmark (replaces experiments/benchmark_gpu.py).

What is measured
  Only the metric evaluation (CASC and CoM) on IDENTICAL packed layouts.
  Packing is performed once, before any timing, and is shared by both paths.
  This isolates the question the GPU code is meant to answer: does batched
  scoring on the GPU beat per-layout scoring on the CPU?

Protocol
  * Warm-up: W untimed calls on each path (CUDA context, kernel setup, JIT).
  * Timed repeats: R calls per path. For CUDA, torch.cuda.synchronize() before
    and after each timed call, because CUDA kernels are asynchronous.
  * Report median and interquartile range over the R repeats.
  * GPU memory: torch.cuda.max_memory_allocated() reset before the timed block.
    Process RSS is reported separately as host memory.

Equivalence (not skipped)
  Per-layout values are compared element by element for every layout that has
  at least one packed item. The check fails if any difference exceeds the
  tolerance; an empty comparison is an error, never a pass.

Usage (repository root):
  python experiments/benchmark_gpu_v2.py --dataset SC1-100 --pop 50 100 250 --out experiments/gpu_v2
"""
import argparse
import json
import os
import platform
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.metrics.casc import CASCv2, CASCv2Config          # noqa: E402
from core.metrics.com import CoMConfig, CoMModel            # noqa: E402
from core.metrics.casc_gpu import CASCv2GPU                 # noqa: E402
from core.metrics.com_gpu import CoMModelGPU                # noqa: E402
from optimization.nsga2.feasibility import decode_layout    # noqa: E402
from optimization.nsga2.operators import CargoSampling      # noqa: E402

TOL = 1e-4
COM_TARGET = np.array([0.5, 0.5, 1.0])


def load_dataset(name):
    path = ROOT / "datasets" / "hyperdock_bench" / f"HB-V1-{name}.json"
    data = json.loads(path.read_text())
    return data["items"], data["containers"][0]["dimensions_wdh_meters"], path


def pack_population(items, container_dims, pop, seed):
    """Pack a fixed random population once. Returns packed layouts with >= 1 item."""
    n = len(items)
    rng = np.random.default_rng(seed)
    layouts = []
    for _ in range(pop):
        seq = rng.permutation(n)
        ori = rng.integers(1, 7, size=n)
        _, _, packed = decode_layout(items, container_dims, seq, ori)
        if len(packed) > 0:
            layouts.append(packed)
    return layouts


def to_padded_tensors(layouts, n_items, device):
    """Padded tensors plus a validity mask, as used by CargoProblemGPU."""
    P = len(layouts)
    pts = np.zeros((P, n_items, 3), dtype=np.float32)
    dims = np.zeros((P, n_items, 3), dtype=np.float32)
    masses = np.zeros((P, n_items), dtype=np.float32)
    cats = np.zeros((P, n_items), dtype=np.float32)
    valid = np.zeros((P, n_items), dtype=bool)
    for i, packed in enumerate(layouts):
        for j, it in enumerate(packed):
            pts[i, j] = it["point"]
            dims[i, j] = it["dimensions"]
            masses[i, j] = it["data"].get("mass_kg", 0.0)
            cats[i, j] = 0.4 if it["data"].get("category", "hardware") in ("consumable", "waste") else 1.0
            valid[i, j] = True
    t = lambda a: torch.from_numpy(a).to(device)  # noqa: E731
    return t(pts), t(dims), t(masses), t(cats), t(valid)


def cpu_score(layouts, casc_cpu, com_cpu):
    casc = np.empty(len(layouts))
    com = np.empty(len(layouts))
    for i, packed in enumerate(layouts):
        casc[i] = casc_cpu.evaluate_layout(packed)["mean_casc"]
        com[i] = com_cpu.evaluate_layout(packed)["delta_com"]
    return casc, com


def timed(fn, sync):
    if sync:
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = fn()
    if sync:
        torch.cuda.synchronize()
    return time.perf_counter() - t0, out


def summarise(times):
    q = statistics.quantiles(times, n=4) if len(times) >= 2 else [times[0], times[0], times[0]]
    return {"median_s": statistics.median(times), "q1_s": q[0], "q3_s": q[2], "repeats": len(times)}


def rss_mb():
    try:
        import psutil
        return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
    except ImportError:
        return float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="SC1-100")
    ap.add_argument("--pop", nargs="+", type=int, default=[50, 100, 250])
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default="experiments/gpu_v2")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sync = device.type == "cuda"
    items, container_dims, ds_path = load_dataset(args.dataset)
    n = len(items)

    casc_cpu = CASCv2(CASCv2Config(grid_resolution=10))
    com_cpu = CoMModel(CoMConfig(target_com=COM_TARGET, dry_mass=5000.0, dry_com=COM_TARGET))
    casc_gpu = CASCv2GPU(CASCv2Config(grid_resolution=10), device=device)
    com_gpu = CoMModelGPU(CoMConfig(target_com=COM_TARGET, dry_mass=5000.0, dry_com=COM_TARGET), device=device)

    env = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(0) if sync else None,
        "dataset": args.dataset, "n_items": n, "dataset_file": str(ds_path.relative_to(ROOT)),
        "warmup": args.warmup, "repeats": args.repeats, "seed": args.seed,
        "timing_scope": "scoring only (CASC and CoM) on identical packed layouts",
    }
    results = []
    for pop in args.pop:
        layouts = pack_population(items, container_dims, pop, args.seed)
        if not layouts:
            raise SystemExit(f"No packable layouts for pop={pop}; cannot benchmark.")
        P = len(layouts)
        pts, dims, masses, cats, valid = to_padded_tensors(layouts, n, device)

        def run_cpu():
            return cpu_score(layouts, casc_cpu, com_cpu)

        def run_gpu():
            with torch.no_grad():
                c = casc_gpu.evaluate_population(pts, dims, masses, cats, valid)
                m = com_gpu.evaluate_population(pts, dims, masses)
            return c.cpu().numpy(), m.cpu().numpy()

        for _ in range(args.warmup):
            run_cpu()
            run_gpu()

        cpu_times = [timed(run_cpu, sync=False)[0] for _ in range(args.repeats)]
        if sync:
            torch.cuda.reset_peak_memory_stats()
        gpu_times = []
        for _ in range(args.repeats):
            dt, _ = timed(run_gpu, sync=sync)
            gpu_times.append(dt)
        gpu_peak_mb = (torch.cuda.max_memory_allocated() / (1024 * 1024)) if sync else float("nan")

        cpu_casc, cpu_com = run_cpu()
        gpu_casc, gpu_com = run_gpu()
        d_casc = float(np.max(np.abs(cpu_casc - gpu_casc)))
        d_com = float(np.max(np.abs(cpu_com - gpu_com)))
        passed = bool(d_casc < TOL and d_com < TOL and P > 0)

        cpu_sum, gpu_sum = summarise(cpu_times), summarise(gpu_times)
        results.append({
            "pop_packable": P,
            "cpu": cpu_sum, "gpu": gpu_sum,
            "speedup_median": cpu_sum["median_s"] / gpu_sum["median_s"],
            "gpu_peak_allocated_mb": gpu_peak_mb,
            "host_rss_mb": rss_mb(),
            "max_abs_diff_casc": d_casc, "max_abs_diff_com": d_com,
            "equivalence_passed": passed,
        })
        print(f"pop={P}: CPU {cpu_sum['median_s']:.3f}s  GPU {gpu_sum['median_s']:.3f}s  "
              f"speedup {results[-1]['speedup_median']:.2f}x  dCASC {d_casc:.2e}  dCoM {d_com:.2e}  "
              f"equiv {passed}", flush=True)

    out_dir = (ROOT / args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "gpu_v2_results.json").write_text(json.dumps({"environment": env, "results": results}, indent=2))

    lines = ["# Scoring-stage CPU vs GPU benchmark", "",
             f"Device: {env['device']} ({env['gpu_name']}), torch {env['torch']} (CUDA {env['torch_cuda']})",
             f"Dataset: {args.dataset} (N={n}); warm-up {args.warmup}, repeats {args.repeats}; "
             "times are medians with IQR; CUDA calls synchronised.", "",
             "| Packable layouts | CPU median (s) | GPU median (s) | Speedup | GPU peak (MB) | max dCASC | max dCoM | Equivalent |",
             "|---|---|---|---|---|---|---|---|"]
    for r in results:
        lines.append(f"| {r['pop_packable']} | {r['cpu']['median_s']:.3f} [{r['cpu']['q1_s']:.3f}, {r['cpu']['q3_s']:.3f}] "
                     f"| {r['gpu']['median_s']:.3f} [{r['gpu']['q1_s']:.3f}, {r['gpu']['q3_s']:.3f}] "
                     f"| {r['speedup_median']:.2f}x | {r['gpu_peak_allocated_mb']:.1f} "
                     f"| {r['max_abs_diff_casc']:.2e} | {r['max_abs_diff_com']:.2e} | {r['equivalence_passed']} |")
    (out_dir / "gpu_v2_report.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
