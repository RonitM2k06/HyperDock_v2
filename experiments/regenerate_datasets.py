import os
import json
import sys
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from core.benchmarks.generator import HyperDockBenchGenerator

def regenerate_all():
    dataset_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'datasets', 'hyperdock_bench'))
    os.makedirs(dataset_dir, exist_ok=True)
    
    scenarios = ["SC1", "SC2", "SC3"]
    scales = [50, 100, 250, 500]
    
    for sc in scenarios:
        for scale in scales:
            bench_id = f"HB-V1-{sc}-{scale}"
            print(f"Generating {bench_id}...")
            
            gen = HyperDockBenchGenerator(bench_id, sc, scale)
            data = gen.generate()
            
            out_path = os.path.join(dataset_dir, f"{bench_id}.json")
            with open(out_path, 'w') as f:
                json.dump(data, f, indent=2)
                
    print("Regeneration complete.")

if __name__ == "__main__":
    regenerate_all()
