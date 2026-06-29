import os
import json
import time
import numpy as np
import pandas as pd
from pathlib import Path
import sys

# Adjust path for core modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.placement.extreme_point import ExtremePointDecoder
from core.placement.rotation import RotationEngine
from optimization.nsga2.problem import CargoProblem
from optimization.nsga2.operators import CargoSampling, CargoCrossover, CargoMutation
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize

class AblationCargoProblem(CargoProblem):
    def __init__(self, items, container_dims, ablation_mode="none"):
        self.ablation_mode = ablation_mode
        super().__init__(items, container_dims)
        
    def _evaluate(self, x, out, *args, **kwargs):
        sequence = x[:self.n_items].astype(int)
        
        # Ablation: Orientation disabled (force default rotation 0)
        if self.ablation_mode == "no_orientation":
            orientations = np.ones(self.n_items, dtype=int)
        else:
            orientations = x[self.n_items:].astype(int)
            
        decoder = ExtremePointDecoder(self.container_dims)
            
        unplaced_volume = 0.0
        unplaced_count = 0
        
        for idx in sequence:
            item_data = self.items[idx]
            base_dims = item_data["dimensions_wdh_meters"]
            rot_idx = orientations[idx] - 1
            all_rots = RotationEngine.all_rotations(base_dims)
            rot_idx = max(0, min(rot_idx, len(all_rots)-1))
            rot_dims = all_rots[rot_idx]
            
            packed = decoder.try_pack_item(rot_dims, item_data)
            
            if not packed:
                v = base_dims[0] * base_dims[1] * base_dims[2]
                unplaced_volume += v
                unplaced_count += 1
                
        packed_items = decoder.packed_items
        
        g1 = unplaced_volume
        g2 = unplaced_count
        
        if len(packed_items) == 0:
            f1, f2 = 9999.0, 0.0
        else:
            if self.ablation_mode == "no_com":
                f1 = 0.0
            else:
                com_res = self.com_evaluator.evaluate_layout(packed_items)
                f1 = com_res['delta_com']
                
            if self.ablation_mode == "no_casc":
                f2 = 0.0
            else:
                casc_res = self.casc_evaluator.evaluate_layout(packed_items)
                f2 = -casc_res['mean_casc']
        
        out["F"] = [f1, f2]
        out["G"] = [g1, g2]

def run_ablation():
    dataset_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'datasets', 'hyperdock_bench', 'HB-V1-SC1-100.json'))
    
    with open(dataset_path, 'r') as f:
        data = json.load(f)
        
    items = data['items']
    n_items = len(items)
    container_dims = data['containers'][0]['dimensions_wdh_meters']
    
    modes = ["none", "no_casc", "no_com", "no_orientation"]
    results = []
    
    for mode in modes:
        print(f"Running Ablation: {mode}")
        for seed in range(1, 6): # 5 runs per mode
            problem = AblationCargoProblem(items=items, container_dims=container_dims, ablation_mode=mode)
            algorithm = NSGA2(
                pop_size=50,
                sampling=CargoSampling(n_items=n_items),
                crossover=CargoCrossover(n_items=n_items, prob=0.9),
                mutation=CargoMutation(n_items=n_items, prob=1.0/n_items),
                eliminate_duplicates=True
            )
            
            start_time = time.time()
            res = minimize(problem, algorithm, ('n_gen', 50), seed=seed, verbose=False)
            runtime = time.time() - start_time
            
            F = res.F
            if F is None:
                F = res.pop.get("F")
            
            X = res.X if res.X is not None else res.pop.get("X")
            
            for x in X[:3]: # Sample up to 3 solutions from the pareto front
                prob_full = AblationCargoProblem(items=items, container_dims=container_dims, ablation_mode="none")
                out = {}
                prob_full._evaluate(x, out)
                true_f1, true_f2 = out["F"]
                true_g1, true_g2 = out["G"]
                
                results.append({
                    "mode": mode,
                    "seed": seed,
                    "runtime": runtime,
                    "unplaced_vol": true_g1,
                    "true_com_dev": true_f1,
                    "true_casc": -true_f2
                })
                
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(os.path.dirname(__file__), 'ablation_results.csv'), index=False)
    print("Ablation study completed.")

if __name__ == "__main__":
    run_ablation()
