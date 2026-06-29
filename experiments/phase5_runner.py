import os
import json
import time
import platform
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from multiprocessing import Pool, cpu_count
from datetime import datetime

# Adjust path for core modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.placement.baselines import FirstFitPacker, BestFitDecreasingVolumePacker, BestFitDecreasingPriorityPacker
from core.placement.extreme_point import ExtremePointDecoder
from core.placement.rotation import RotationEngine
from core.metrics.casc import CASCv2, CASCv2Config
from core.metrics.com import CoMModel, CoMConfig
from optimization.nsga2.problem import CargoProblem
from optimization.nsga2.problem_gpu import CargoProblemGPU
from optimization.nsga2.operators import CargoSampling, CargoCrossover, CargoMutation
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.core.callback import Callback

import importlib.metadata

class ConvergenceLogger(Callback):
    def __init__(self, run_id):
        super().__init__()
        self.run_id = run_id
        self.history = []

    def notify(self, algorithm):
        gen = algorithm.n_gen
        F = algorithm.pop.get("F")
        CV = algorithm.pop.get("CV")
        
        cv_min = CV.min() if CV is not None else 0
        
        if CV is not None:
            feasible_mask = (CV <= 0).flatten()
            feasible = F[feasible_mask] if np.any(feasible_mask) else F
        else:
            feasible = F
            
        if len(feasible) > 0:
            best_f1 = feasible[:, 0].min()
            best_f2 = feasible[:, 1].min()
            mean_f1 = feasible[:, 0].mean()
            mean_f2 = feasible[:, 1].mean()
        else:
            best_f1, best_f2, mean_f1, mean_f2 = None, None, None, None
            
        self.history.append({
            "run_id": self.run_id,
            "generation": gen,
            "cv_min": cv_min,
            "best_com_dev": best_f1,
            "best_neg_casc": best_f2,
            "mean_com_dev": mean_f1,
            "mean_neg_casc": mean_f2
        })

def evaluate_layout(packed_items):
    if not packed_items:
        return 9999.0, 0.0
        
    com_eval = CoMModel(CoMConfig(
        target_com=np.array([0.5, 0.5, 1.0]),
        dry_mass=5000.0,
        dry_com=np.array([0.5, 0.5, 1.0])
    ))
    casc_eval = CASCv2(CASCv2Config(grid_resolution=10))
    
    com_res = com_eval.evaluate_layout(packed_items)
    casc_res = casc_eval.evaluate_layout(packed_items)
    
    return com_res['delta_com'], casc_res['mean_casc']

def run_baseline(args):
    dataset_path, algo_name = args
    dataset_name = os.path.basename(dataset_path).replace(".json", "")
    
    with open(dataset_path, 'r') as f:
        data = json.load(f)
        
    items = data['items']
    container_dims = data['containers'][0]['dimensions_wdh_meters']
    total_cargo_vol = sum([i['dimensions_wdh_meters'][0]*i['dimensions_wdh_meters'][1]*i['dimensions_wdh_meters'][2] for i in items])
    container_vol = container_dims[0] * container_dims[1] * container_dims[2]
    
    if algo_name == "FF":
        packer = FirstFitPacker(container_dims)
    elif algo_name == "BFD-V":
        packer = BestFitDecreasingVolumePacker(container_dims)
    else:
        packer = BestFitDecreasingPriorityPacker(container_dims)
        
    start_time = time.time()
    packed_count, packed_volume = packer.pack(items)
    runtime = time.time() - start_time
    
    com_dev, casc_score = evaluate_layout(packer.decoder.packed_items)
    utilization = (packed_volume / container_vol) * 100 if container_vol > 0 else 0
    
    return {
        "dataset": dataset_name,
        "algorithm": algo_name,
        "seed": 0,
        "runtime_s": runtime,
        "packed_items": packed_count,
        "total_items": len(items),
        "packed_volume_m3": packed_volume,
        "total_cargo_vol_m3": total_cargo_vol,
        "container_vol_m3": container_vol,
        "utilization_pct": utilization,
        "com_deviation": com_dev,
        "casc_score": casc_score,
        "generations": 0,
        "pareto_size": 1
    }

def run_nsga2(args):
    dataset_path, seed, _ = args
    dataset_name = os.path.basename(dataset_path).replace(".json", "")
    run_id = f"{dataset_name}_seed_{seed}"
    
    with open(dataset_path, 'r') as f:
        data = json.load(f)
        
    items = data['items']
    n_items = len(items)
    container_dims = data['containers'][0]['dimensions_wdh_meters']
    total_cargo_vol = sum([i['dimensions_wdh_meters'][0]*i['dimensions_wdh_meters'][1]*i['dimensions_wdh_meters'][2] for i in items])
    container_vol = container_dims[0] * container_dims[1] * container_dims[2]
    
    problem = CargoProblemGPU(items=items, container_dims=container_dims, n_workers=0)
    
    # Limit population and generations for extremely large datasets to maintain tractability in this demo phase
    pop_size = 50 if n_items >= 250 else 100
    n_gen = 50 if n_items >= 250 else 100
    
    algorithm = NSGA2(
        pop_size=pop_size,
        sampling=CargoSampling(n_items=n_items),
        crossover=CargoCrossover(n_items=n_items, prob=0.9),
        mutation=CargoMutation(n_items=n_items, prob=1.0/n_items),
        eliminate_duplicates=False
    )
    
    logger = ConvergenceLogger(run_id)
    
    start_time = time.time()
    res = minimize(
        problem,
        algorithm,
        ('n_gen', n_gen),
        seed=seed,
        callback=logger,
        verbose=False
    )
    runtime = time.time() - start_time
    
    # Process Pareto front
    F = res.F
    X = res.X
    pareto_size = len(F) if F is not None else 0
    
    if pareto_size == 0:
        F = res.pop.get("F")
        X = res.pop.get("X")
        pareto_size = len(F)
        
    # Save Pareto front
    pareto_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'pareto_fronts'))
    os.makedirs(pareto_dir, exist_ok=True)
    pd.DataFrame(F, columns=['com_deviation', 'neg_casc']).to_csv(os.path.join(pareto_dir, f"{run_id}.csv"), index=False)
    
    # Re-evaluate best compromised solution for packed volume & count
    best_f1, best_f2 = 9999.0, 0.0
    packed_count, packed_vol = 0, 0.0
    
    if pareto_size > 0:
        f_min = F.min(axis=0)
        f_max = F.max(axis=0)
        denom = np.where(f_max - f_min == 0, 1, f_max - f_min)
        norm_F = (F - f_min) / denom
        dists = np.linalg.norm(norm_F, axis=1)
        best_idx = np.argmin(dists)
        
        best_f1 = F[best_idx, 0]
        best_f2 = F[best_idx, 1]
        
        best_x = X[best_idx]
        sequence = best_x[:n_items].astype(int)
        orientations = best_x[n_items:].astype(int)
        
        decoder = ExtremePointDecoder(container_dims)
        for idx in sequence:
            item_data = items[idx]
            base_dims = item_data["dimensions_wdh_meters"]
            rot_idx = orientations[idx] - 1
            all_rots = RotationEngine.all_rotations(base_dims)
            rot_idx = max(0, min(rot_idx, len(all_rots)-1))
            rot_dims = all_rots[rot_idx]
            if decoder.try_pack_item(rot_dims, item_data):
                packed_vol += (rot_dims[0] * rot_dims[1] * rot_dims[2])
                
        packed_count = len(decoder.packed_items)
        
    utilization = (packed_vol / container_vol) * 100 if container_vol > 0 else 0
        
    return {
        "result": {
            "dataset": dataset_name,
            "algorithm": "NSGA-II",
            "seed": seed,
            "runtime_s": runtime,
            "packed_items": packed_count,
            "total_items": len(items),
            "packed_volume_m3": packed_vol,
            "total_cargo_vol_m3": total_cargo_vol,
            "container_vol_m3": container_vol,
            "utilization_pct": utilization,
            "com_deviation": best_f1,
            "casc_score": -best_f2,
            "generations": n_gen,
            "pareto_size": pareto_size
        },
        "convergence": logger.history
    }

def collect_metadata():
    repro_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'reproducibility'))
    os.makedirs(repro_dir, exist_ok=True)
    
    packages = ["numpy", "pandas", "pymoo", "scipy"]
    pkg_versions = {}
    for p in packages:
        try:
            pkg_versions[p] = importlib.metadata.version(p)
        except:
            pkg_versions[p] = "unknown"
            
    meta = {
        "timestamp": datetime.now().isoformat(),
        "python_version": platform.python_version(),
        "cpu_info": platform.processor(),
        "os": platform.platform(),
        "logical_cores": cpu_count(),
        "packages": pkg_versions
    }
    
    with open(os.path.join(repro_dir, 'metadata.json'), 'w') as f:
        json.dump(meta, f, indent=4)

def main():
    print("Starting Phase 5: Large Scale Experiments")
    collect_metadata()
    
    dataset_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'datasets', 'hyperdock_bench'))
    datasets = [os.path.join(dataset_dir, f) for f in os.listdir(dataset_dir) if f.endswith('.json')]
    
    baseline_args = []
    nsga2_args = []
    
    for ds in datasets:
        for algo in ["FF", "BFD-V", "BFD-P"]:
            baseline_args.append((ds, algo))
        for seed in range(1, 31):
            nsga2_args.append((ds, seed, seed))
            
    print(f"Total datasets: {len(datasets)}")
    print(f"Total baseline tasks: {len(baseline_args)}")
    print(f"Total NSGA-II tasks: {len(nsga2_args)}")
    
    results = []
    convergence_data = []
    
    # Run baselines
    print("Executing Baseline algorithms sequentially...")
    for arg in baseline_args:
        results.append(run_baseline(arg))
        
    # Run NSGA-II in parallel
    workers = min(4, cpu_count())
    print(f"Executing NSGA-II with {workers} workers to conserve memory...")
    start_time = time.time()
    with Pool(processes=workers) as pool:
        for i, out in enumerate(pool.imap_unordered(run_nsga2, nsga2_args)):
            results.append(out["result"])
            convergence_data.extend(out["convergence"])
            
            res = out["result"]
            dataset = res["dataset"]
            seed = res["seed"]
            print(f"[{i+1}/{len(nsga2_args)}] Completed NSGA-II run for {dataset} (Seed {seed})")
                
    print(f"All experiments completed in {time.time() - start_time:.2f} seconds.")
    
    # Save outputs
    res_df = pd.DataFrame(results)
    res_df.to_csv(os.path.join(os.path.dirname(__file__), 'results.csv'), index=False)
    
    conv_df = pd.DataFrame(convergence_data)
    conv_df.to_csv(os.path.join(os.path.dirname(__file__), 'convergence.csv'), index=False)
    print("Saved results.csv and convergence.csv")

if __name__ == "__main__":
    main()
