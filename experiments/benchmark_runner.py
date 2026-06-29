import time
import os
import json
import csv
from core.benchmarks.generator import HyperDockBenchGenerator
from core.benchmarks.validator import HyperDockBenchValidator
from core.placement.baselines import FirstFitPacker, BestFitDecreasingVolumePacker, BestFitDecreasingPriorityPacker

def run_benchmarks():
    scenarios = ["SC1", "SC2", "SC3"]
    scales = [50, 100, 250, 500]
    
    dataset_dir = os.path.join("datasets", "hyperdock_bench")
    os.makedirs(dataset_dir, exist_ok=True)
    
    results = []

    for sc in scenarios:
        for scale in scales:
            bench_id = f"HB-V1-{sc}-{scale}"
            print(f"\n--- Generating and Running {bench_id} ---")
            
            # Generate
            gen = HyperDockBenchGenerator(bench_id, sc, scale)
            data = gen.generate()
            
            # Validate
            if not HyperDockBenchValidator.validate_all(data):
                print(f"Skipping {bench_id} due to validation failure.")
                continue
                
            # Save dataset
            file_path = os.path.join(dataset_dir, f"{bench_id}.json")
            with open(file_path, "w") as f:
                json.dump(data, f, indent=2)
                
            # Calculate total container volume (assuming 1 container for simplicity in baselines, 
            # but generator might create multiple. We'll sum them.)
            total_container_vol = sum(
                [c["dimensions_wdh_meters"][0]*c["dimensions_wdh_meters"][1]*c["dimensions_wdh_meters"][2] 
                 for c in data["containers"]]
            )
            
            # For baselines, we'll assume we pack into the FIRST container for simplicity in this MVP 
            # (or we could pack into a single theoretical unified volume).
            # The generator currently outputs multiple A, B, C containers. 
            # Let's pack into the first one and see how many fit, OR we can aggregate them.
            # To keep the baseline simple, let's treat the total volume as a single large bin for the exact dimensions 
            # of the first container if there's only one, else we'll just pack into Container 1.
            
            # Let's use the first container
            c1 = data["containers"][0]
            c1_dims = c1["dimensions_wdh_meters"]
            c1_vol = c1_dims[0] * c1_dims[1] * c1_dims[2]
            
            items = data["items"]
            
            algorithms = {
                "FF": FirstFitPacker(c1_dims),
                "BFD-V": BestFitDecreasingVolumePacker(c1_dims),
                "BFD-P": BestFitDecreasingPriorityPacker(c1_dims)
            }
            
            for alg_name, packer in algorithms.items():
                start_time = time.perf_counter()
                packed_count, packed_vol = packer.pack(items)
                elapsed = time.perf_counter() - start_time
                
                utilization = packed_vol / c1_vol
                success_rate = packed_count / len(items)
                
                results.append({
                    "Benchmark": bench_id,
                    "Algorithm": alg_name,
                    "Total_Items": len(items),
                    "Packed_Items": packed_count,
                    "Container_Vol": round(c1_vol, 3),
                    "Packed_Vol": round(packed_vol, 3),
                    "Utilization": round(utilization, 4),
                    "Time_s": round(elapsed, 4)
                })
                
                print(f"{alg_name:6s} | Packed: {packed_count}/{len(items)} | Util: {utilization*100:.1f}% | Time: {elapsed:.4f}s")

    # Output CSV
    csv_file = os.path.join("experiments", "baseline_results.csv")
    with open(csv_file, "w", newline='') as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"\nResults saved to {csv_file}")

if __name__ == "__main__":
    run_benchmarks()
