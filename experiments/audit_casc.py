import json
import csv
import sys
import os
import glob
import numpy as np
from pathlib import Path

# Ensure core modules can be imported
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.placement.baselines import (
    FirstFitPacker, 
    BestFitDecreasingVolumePacker, 
    BestFitDecreasingPriorityPacker
)
from core.metrics.casc import CASCv2, CASCv2Config

def run_audit(dataset_dir: str, output_csv: str):
    json_files = glob.glob(os.path.join(dataset_dir, 'HB-V1-*.json'))
    json_files.sort()
    
    if not json_files:
        print(f"No JSON files found in {dataset_dir}")
        return

    casc_evaluator = CASCv2(CASCv2Config(grid_resolution=20))
    
    results = []
    
    for filepath in json_files:
        dataset_name = os.path.basename(filepath)
        print(f"Auditing: {dataset_name}")
        with open(filepath, 'r') as f:
            data = json.load(f)
            
        items = data['items']
        container_dims = data['containers'][0]['dimensions_wdh_meters']
        
        packers = {
            'FirstFit': FirstFitPacker(container_dims),
            'BFD-Volume': BestFitDecreasingVolumePacker(container_dims),
            'BFD-Priority': BestFitDecreasingPriorityPacker(container_dims)
        }
        
        for algo_name, packer in packers.items():
            packed_count, packed_volume = packer.pack(items)
            packed_items = packer.decoder.packed_items
            
            casc_result = casc_evaluator.evaluate_layout(packed_items)
            scores = [d['casc_score'] for d in casc_result['item_details']]
            
            if scores:
                mean_casc = np.mean(scores)
                min_casc = np.min(scores)
                max_casc = np.max(scores)
                std_casc = np.std(scores)
            else:
                mean_casc = min_casc = max_casc = std_casc = 0.0
                
            results.append({
                'Dataset': dataset_name,
                'Algorithm': algo_name,
                'Mean_CASC': f"{mean_casc:.4f}",
                'Min_CASC': f"{min_casc:.4f}",
                'Max_CASC': f"{max_casc:.4f}",
                'Std_CASC': f"{std_casc:.4f}"
            })
            
    # Write to CSV
    output_path = Path(__file__).parent / output_csv
    with open(output_path, 'w', newline='') as csvfile:
        fieldnames = ['Dataset', 'Algorithm', 'Mean_CASC', 'Min_CASC', 'Max_CASC', 'Std_CASC']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for row in results:
            writer.writerow(row)
            
    print(f"\nAudit complete. Results written to {output_path}")

if __name__ == "__main__":
    dataset_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'datasets', 'hyperdock_bench'))
    run_audit(dataset_dir, 'casc_audit_results.csv')
