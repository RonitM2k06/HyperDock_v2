import json
import os
import glob
from pathlib import Path

def audit_datasets(dataset_dir: str):
    json_files = glob.glob(os.path.join(dataset_dir, 'HB-V1-*.json'))
    json_files.sort()
    
    if not json_files:
        print(f"No JSON files found in {dataset_dir}")
        return

    print(f"{'Dataset':<20} | {'Cont Vol (m3)':<15} | {'Cargo Vol (m3)':<15} | {'Util (%)':<10} | {'Feasible':<10} | {'Status'}")
    print("-" * 90)
    
    for filepath in json_files:
        dataset_name = os.path.basename(filepath)
        with open(filepath, 'r') as f:
            data = json.load(f)
            
        items = data.get('items', [])
        containers = data.get('containers', [])
        
        container_vol = 0.0
        for c in containers:
            dims = c['dimensions_wdh_meters']
            container_vol += dims[0] * dims[1] * dims[2]
            
        cargo_vol = 0.0
        for item in items:
            dims = item['dimensions_wdh_meters']
            cargo_vol += dims[0] * dims[1] * dims[2]
            
        util = (cargo_vol / container_vol) * 100 if container_vol > 0 else 0
        
        feasible = "Yes" if util <= 100.0 else "No"
        
        # Check against spec
        status = "OK"
        if "SC1" in dataset_name and not (75 <= util <= 85):
            status = "Violates SC1 spec (~80%)"
        elif "SC3" in dataset_name and not (94 <= util <= 98):
            status = "Violates SC3 spec (~96%)"
        if util > 100.0:
            status = "INFEASIBLE (>100%)"
            
        print(f"{dataset_name:<20} | {container_vol:<15.4f} | {cargo_vol:<15.4f} | {util:<10.2f} | {feasible:<10} | {status}")

if __name__ == "__main__":
    dataset_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'datasets', 'hyperdock_bench'))
    audit_datasets(dataset_dir)
