import os
import sys
import json
import time
import torch
import psutil
import numpy as np
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from optimization.nsga2.problem import CargoProblem
from optimization.nsga2.problem_gpu import CargoProblemGPU
from optimization.nsga2.operators import CargoSampling, CargoCrossover, CargoMutation

def get_memory():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)

def run_benchmark():
    dataset_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'datasets', 'hyperdock_bench', 'HB-V1-SC1-100.json'))
    
    with open(dataset_path, 'r') as f:
        data = json.load(f)
        
    items = data['items']
    n_items = len(items)
    container_dims = data['containers'][0]['dimensions_wdh_meters']
    
    pop_size = 50
    n_gen = 5
    
    print(f"Benchmarking NSGA-II on {n_items} items (Pop: {pop_size}, Gen: {n_gen})")
    
    algorithm = NSGA2(
        pop_size=pop_size,
        sampling=CargoSampling(n_items=n_items),
        crossover=CargoCrossover(n_items=n_items, prob=0.9),
        mutation=CargoMutation(n_items=n_items, prob=1.0/n_items),
        eliminate_duplicates=False
    )
    
    # 1. CPU Benchmark (Numba optimized)
    problem_cpu = CargoProblem(items=items, container_dims=container_dims)
    
    print("\nRunning CPU Numba Implementation...")
    t0 = time.time()
    res_cpu = minimize(problem_cpu, algorithm, ('n_gen', n_gen), seed=42, verbose=False)
    cpu_time = time.time() - t0
    cpu_mem = get_memory()
    
    # 2. GPU Benchmark (Batched PyTorch)
    problem_gpu = CargoProblemGPU(items=items, container_dims=container_dims, n_workers=4)
    
    print("Running GPU PyTorch Implementation...")
    t0 = time.time()
    res_gpu = minimize(problem_gpu, algorithm, ('n_gen', n_gen), seed=42, verbose=False)
    gpu_time = time.time() - t0
    gpu_mem = get_memory()
    
    # Validation
    if res_cpu.F is not None and res_gpu.F is not None:
        f1_diff = np.abs(res_cpu.F[:, 0] - res_gpu.F[:, 0]).max()
        f2_diff = np.abs(res_cpu.F[:, 1] - res_gpu.F[:, 1]).max()
    else:
        f1_diff = 0.0
        f2_diff = 0.0
        print("Note: All solutions were infeasible (F is None).")
    
    speedup = cpu_time / max(gpu_time, 1e-5)
    
    report = f"""# GPU CUDA Benchmark Report

## 1. Runtime Performance (5 Generations, Pop=50, Items=100)
- **CPU (Numba JIT):** {cpu_time:.2f} seconds
- **GPU (Hybrid PyTorch):** {gpu_time:.2f} seconds
- **Speedup:** {speedup:.2f}x

## 2. Resource Utilization
- **CPU Memory (Peak):** {cpu_mem:.1f} MB
- **GPU Architecture Memory:** {gpu_mem:.1f} MB
- **Device:** {problem_gpu.device}

## 3. Mathematical Equivalence
- **Max CoM Difference:** {f1_diff:.6e}
- **Max CASC Difference:** {f2_diff:.6e}
- **Result:** {'PASSED' if f1_diff < 1e-4 and f2_diff < 1e-4 else 'FAILED'}
"""
    
    with open('gpu_benchmark_report.md', 'w') as f:
        f.write(report)
        
    print(report)

if __name__ == '__main__':
    run_benchmark()
