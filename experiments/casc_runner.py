import json
import csv
import sys
import os
from pathlib import Path

# Ensure core modules can be imported
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.placement.baselines import (
    FirstFitPacker, 
    BestFitDecreasingVolumePacker, 
    BestFitDecreasingPriorityPacker
)
from core.metrics.casc import CASCv2, CASCv2Config

def run_casc_experiment(dataset_path: str, output_csv: str):
    print(f"Loading dataset from: {dataset_path}")
    with open(dataset_path, 'r') as f:
        data = json.load(f)
    
    items = data['items']
    # Just take the first container dimensions for simplicity
    container_dims = data['containers'][0]['dimensions_wdh_meters']
    
    packers = {
        'FirstFit': FirstFitPacker(container_dims),
        'BFD-Volume': BestFitDecreasingVolumePacker(container_dims),
        'BFD-Priority': BestFitDecreasingPriorityPacker(container_dims)
    }
    
    casc_evaluator = CASCv2(CASCv2Config(grid_resolution=20))
    
    results = []
    
    for name, packer in packers.items():
        print(f"Running packer: {name}")
        packed_count, packed_volume = packer.pack(items)
        packed_items = packer.decoder.packed_items
        
        print(f"  Packed {packed_count}/{len(items)} items. Volume: {packed_volume:.2f} m3")
        
        # Evaluate CASC
        casc_result = casc_evaluator.evaluate_layout(packed_items)
        mean_casc = casc_result['mean_casc']
        total_casc = casc_result['total_casc']
        
        print(f"  Mean CASC v2: {mean_casc:.4f}")
        
        for detail in casc_result['item_details']:
            results.append({
                'Algorithm': name,
                'ItemID': detail['item_id'],
                'CASC_Score': f"{detail['casc_score']:.4f}",
                'Reach_Penalty': f"{detail['c_reach']:.4f}",
                'Occlusion_Penalty': f"{detail['c_occ']:.4f}",
                'Extraction_Penalty': f"{detail['c_ext']:.4f}",
                'Blocker_Count': detail['transitive_blockers_count']
            })
            
    # Write to CSV
    output_path = Path(__file__).parent / output_csv
    with open(output_path, 'w', newline='') as csvfile:
        fieldnames = ['Algorithm', 'ItemID', 'CASC_Score', 'Reach_Penalty', 'Occlusion_Penalty', 'Extraction_Penalty', 'Blocker_Count']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for row in results:
            writer.writerow(row)
            
    print(f"Results successfully written to {output_path}")

if __name__ == "__main__":
    dataset = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'datasets', 'hyperdock_bench', 'HB-V1-SC1-50.json'))
    run_casc_experiment(dataset, 'casc_results.csv')
