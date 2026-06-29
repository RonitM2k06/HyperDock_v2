import cProfile
import pstats
import os
import sys
import json
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from optimization.nsga2.problem import CargoProblem
from optimization.nsga2.operators import CargoSampling, CargoCrossover, CargoMutation
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize

def run_profile():
    dataset_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'datasets', 'hyperdock_bench', 'HB-V1-SC1-100.json'))
    
    with open(dataset_path, 'r') as f:
        data = json.load(f)
        
    items = data['items']
    n_items = len(items)
    container_dims = data['containers'][0]['dimensions_wdh_meters']
    
    problem = CargoProblem(items=items, container_dims=container_dims)
    algorithm = NSGA2(
        pop_size=20, # Small pop size for profiling
        sampling=CargoSampling(n_items=n_items),
        crossover=CargoCrossover(n_items=n_items, prob=0.9),
        mutation=CargoMutation(n_items=n_items, prob=1.0/n_items),
        eliminate_duplicates=False
    )
    
    print("Starting profile run...")
    start_time = time.time()
    res = minimize(problem, algorithm, ('n_gen', 5), seed=1, verbose=False)
    print(f"Profile run completed in {time.time() - start_time:.2f} seconds.")

if __name__ == '__main__':
    profiler = cProfile.Profile()
    profiler.enable()
    run_profile()
    profiler.disable()
    
    stats = pstats.Stats(profiler).sort_stats('cumtime')
    stats.dump_stats('profile_results.prof')
    
    with open('profile_summary.txt', 'w') as f:
        sys.stdout = f
        stats.print_stats(50)
        sys.stdout = sys.__stdout__
    print("Profile saved to profile_summary.txt")
