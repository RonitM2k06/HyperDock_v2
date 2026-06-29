import json
import os
import sys
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from optimization.nsga2.problem import CargoProblem
from pymoo.core.sampling import Sampling

class CargoSampling(Sampling):
    def _do(self, problem, n_samples, **kwargs):
        X = np.zeros((n_samples, problem.n_var))
        for i in range(n_samples):
            X[i, :problem.n_items] = np.random.permutation(problem.n_items)
            X[i, problem.n_items:] = np.random.randint(1, 7, size=problem.n_items)
        return X

def run_scaling_audit(dataset_path: str):
    print(f"Loading dataset: {dataset_path}")
    with open(dataset_path, 'r') as f:
        data = json.load(f)
        
    items = data['items']
    container_dims = data['containers'][0]['dimensions_wdh_meters']
    
    problem = CargoProblem(items=items, container_dims=container_dims)
    sampler = CargoSampling()
    
    # Generate 100 random chromosomes
    X = sampler._do(problem, 100)
    
    results = []
    
    feasible_count = 0
    penalized_count = 0
    total_unpacked = 0
    
    # Evaluate manually to capture both raw and penalized
    for i in range(100):
        x = X[i]
        
        sequence = x[:problem.n_items].astype(int)
        orientations = x[problem.n_items:].astype(int)
        
        from core.placement.extreme_point import ExtremePointDecoder
        from core.placement.rotation import RotationEngine
        
        decoder = ExtremePointDecoder(problem.container_dims)
        
        unplaced_volume = 0.0
        unplaced_mass = 0.0
        unplaced_count = 0
        
        for idx in sequence:
            item_data = problem.items[idx]
            base_dims = item_data["dimensions_wdh_meters"]
            rot_idx = orientations[idx] - 1
            all_rots = RotationEngine.all_rotations(base_dims)
            rot_idx = max(0, min(rot_idx, len(all_rots)-1))
            rot_dims = all_rots[rot_idx]
            
            packed = decoder.try_pack_item(rot_dims, item_data)
            if not packed:
                v = base_dims[0] * base_dims[1] * base_dims[2]
                unplaced_volume += v
                unplaced_mass += item_data.get('mass_kg', 0.0)
                unplaced_count += 1
                
        packed_items = decoder.packed_items
        
        # Raw metrics
        com_res = problem.com_evaluator.evaluate_layout(packed_items)
        raw_com = com_res['delta_com']
        
        casc_res = problem.casc_evaluator.evaluate_layout(packed_items)
        raw_casc = casc_res['mean_casc']
        
        # Penalized metrics
        penalized_com = raw_com + problem.lambda_pen * unplaced_mass
        penalized_casc = -raw_casc + problem.eta_pen * unplaced_count
        
        if unplaced_count == 0:
            feasible_count += 1
        else:
            penalized_count += 1
            
        total_unpacked += unplaced_count
        
        results.append({
            'unplaced_vol': unplaced_volume,
            'raw_com': raw_com,
            'raw_casc': raw_casc,
            'pen_com': penalized_com,
            'pen_casc': penalized_casc,
            'unpacked_count': unplaced_count
        })

    print(f"\n--- Feasibility Analysis ---")
    print(f"Feasible Solutions (0 items unpacked): {feasible_count} / 100 ({feasible_count}%)")
    print(f"Penalized Solutions (>0 items unpacked): {penalized_count} / 100 ({penalized_count}%)")
    print(f"Average Unpacked Items: {total_unpacked / 100.0:.2f} (out of {problem.n_items})")

    print(f"\n--- Objective Scaling Summary (Means) ---")
    avg_uvol = np.mean([r['unplaced_vol'] for r in results])
    avg_raw_com = np.mean([r['raw_com'] for r in results])
    avg_raw_casc = np.mean([r['raw_casc'] for r in results])
    avg_pen_com = np.mean([r['pen_com'] for r in results])
    avg_pen_casc = np.mean([r['pen_casc'] for r in results])
    
    print(f"Avg Unplaced Volume: {avg_uvol:.4f} m^3")
    print(f"Avg Raw CoM Dev:     {avg_raw_com:.4f} m")
    print(f"Avg Raw CASC:        {avg_raw_casc:.4f}")
    print(f"Avg Penalized CoM:   {avg_pen_com:.4f}")
    print(f"Avg Penalized CASC:  {avg_pen_casc:.4f}")
    
    print("\n--- Example Raw vs Penalized ---")
    for i in range(3):
        r = results[i]
        print(f"Sample {i+1}: Unpacked={r['unpacked_count']} | Raw CoM={r['raw_com']:.4f} -> Pen CoM={r['pen_com']:.4f} | Raw CASC={r['raw_casc']:.4f} -> Pen CASC={r['pen_casc']:.4f}")

if __name__ == "__main__":
    dataset = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'datasets', 'hyperdock_bench', 'HB-V1-SC1-50.json'))
    run_scaling_audit(dataset)
