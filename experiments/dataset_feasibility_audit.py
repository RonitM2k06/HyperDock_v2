import json
import os
import glob
import csv

def audit_all_datasets(dataset_dir: str):
    print("Auditing all datasets in:", dataset_dir)
    files = glob.glob(os.path.join(dataset_dir, '*.json'))
    
    results = []
    
    for fpath in files:
        with open(fpath, 'r') as f:
            data = json.load(f)
            
        dataset_name = os.path.basename(fpath)
        items = data['items']
        container = data['containers'][0]
        cdims = container['dimensions_wdh_meters']
        
        container_vol = cdims[0] * cdims[1] * cdims[2]
        
        cargo_vol = 0.0
        for item in items:
            idims = item['dimensions_wdh_meters']
            cargo_vol += idims[0] * idims[1] * idims[2]
            
        utilization = (cargo_vol / container_vol) * 100.0
        
        feasible = "NO" if utilization > 100.0 else "YES"
        if utilization > 85.0 and feasible == "YES":
            feasible = "YES (Highly Constrained)"
            
        results.append({
            'Dataset': dataset_name,
            'Container_Volume': container_vol,
            'Cargo_Volume': cargo_vol,
            'Required_Density': utilization,
            'Feasible_YN': feasible
        })
        
    results.sort(key=lambda x: x['Dataset'])
    
    print(f"{'Dataset':<20} | {'Container_Vol':<15} | {'Cargo_Vol':<15} | {'Required_Density':<20} | {'Feasible_YN'}")
    print("-" * 95)
    for r in results:
        print(f"{r['Dataset']:<20} | {r['Container_Volume']:<15.4f} | {r['Cargo_Volume']:<15.4f} | {r['Required_Density']:<19.2f}% | {r['Feasible_YN']}")
        
    return results

if __name__ == "__main__":
    dataset_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'datasets', 'hyperdock_bench'))
    audit_all_datasets(dataset_dir)
