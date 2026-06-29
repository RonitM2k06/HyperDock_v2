import json
import csv
import sys
import os
import numpy as np
from pathlib import Path
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize

# Ensure core modules can be imported
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from optimization.nsga2 import CargoProblem, CargoCrossover, CargoMutation, CargoSampling

def run_nsga2_experiment(dataset_path: str):
    print(f"Loading dataset: {dataset_path}")
    with open(dataset_path, 'r') as f:
        data = json.load(f)
        
    items = data['items']
    container_dims = data['containers'][0]['dimensions_wdh_meters']
    n_items = len(items)
    
    print(f"Setting up NSGA-II for {n_items} items.")
    problem = CargoProblem(items=items, container_dims=container_dims)
    
    algorithm = NSGA2(
        pop_size=100,
        sampling=CargoSampling(n_items=n_items),
        crossover=CargoCrossover(n_items=n_items, prob=0.9),
        mutation=CargoMutation(n_items=n_items, prob=1.0/n_items),
        eliminate_duplicates=True
    )
    
    print("Starting optimization... (This may take a minute)")
    res = minimize(
        problem,
        algorithm,
        ('n_gen', 100),
        seed=1,
        verbose=True
    )
    
    # Process results
    F = res.F
    X = res.X
    
    pareto_size = len(F) if F is not None else 0
    print(f"\nOptimization complete.")
    print(f"Pareto front size (feasible): {pareto_size}")
    
    if pareto_size == 0:
        print("No completely feasible solutions found. Extracting best from final population.")
        F = res.pop.get("F")
        X = res.pop.get("X")
        pareto_size = len(F)
        
    # Find extremes
    best_f1_idx = np.argmin(F[:, 0]) # Min CoM Deviation
    best_f2_idx = np.argmin(F[:, 1]) # Min -CASC (Max CASC)
    
    print("\nBest Solutions (Objectives: [CoM Dev, -CASC]):")
    print(f"Best by CoM Deviation:   {F[best_f1_idx]}")
    print(f"Best by CASC Score:      {F[best_f2_idx]}")
    
    # Save pareto_front.csv
    base_dir = Path(__file__).parent
    
    pareto_path = base_dir / 'pareto_front.csv'
    with open(pareto_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['CoM_Deviation', 'Negative_CASC'])
        for obj in F:
            writer.writerow(obj)
            
    # Save optimization_results.csv (including sequence and orientation)
    results_path = base_dir / 'optimization_results.csv'
    with open(results_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Solution_ID', 'CoM_Deviation', 'Negative_CASC', 'Sequence', 'Orientation'])
        for i in range(pareto_size):
            seq = X[i, :n_items].tolist()
            ori = X[i, n_items:].tolist()
            writer.writerow([i, F[i, 0], F[i, 1], str(seq), str(ori)])
            
    print(f"\nOutputs saved to {pareto_path} and {results_path}")

if __name__ == "__main__":
    dataset_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'datasets', 'hyperdock_bench', 'HB-V1-SC1-50.json'))
    run_nsga2_experiment(dataset_path)
