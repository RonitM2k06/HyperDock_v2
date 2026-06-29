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
from core.metrics.com import CoMModel, CoMConfig

def run_com_experiment(dataset_dir: str, output_csv: str):
    json_files = glob.glob(os.path.join(dataset_dir, 'HB-V1-*.json'))
    json_files.sort()
    
    if not json_files:
        print(f"No JSON files found in {dataset_dir}")
        return

    # Define a realistic spacecraft dry mass and a target CoM
    config = CoMConfig(
        target_com=np.array([0.5, 0.5, 1.0]),
        dry_mass=5000.0,
        dry_com=np.array([0.5, 0.5, 1.0]),
        epsilon_x=0.03,
        epsilon_y=0.10,
        epsilon_z=0.10
    )
    com_evaluator = CoMModel(config)
    
    results = []
    
    for filepath in json_files:
        dataset_name = os.path.basename(filepath)
        print(f"Evaluating CoM for: {dataset_name}")
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
            
            com_result = com_evaluator.evaluate_layout(packed_items)
            
            sys_com = com_result['system_com']
            dev = com_result['delta_com']
            ellip_dev = com_result['ellipsoidal_dev']
            
            results.append({
                'Dataset': dataset_name,
                'Algorithm': algo_name,
                'CoM_X': f"{sys_com[0]:.4f}",
                'CoM_Y': f"{sys_com[1]:.4f}",
                'CoM_Z': f"{sys_com[2]:.4f}",
                'CoM_Deviation': f"{dev:.4f}",
                'Ellipsoidal_Deviation': f"{ellip_dev:.4f}"
            })
            
    # Write to CSV
    output_path = Path(__file__).parent / output_csv
    with open(output_path, 'w', newline='') as csvfile:
        fieldnames = ['Dataset', 'Algorithm', 'CoM_X', 'CoM_Y', 'CoM_Z', 'CoM_Deviation', 'Ellipsoidal_Deviation']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for row in results:
            writer.writerow(row)
            
    print(f"\nCoM experiments complete. Results written to {output_path}")

if __name__ == "__main__":
    dataset_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'datasets', 'hyperdock_bench'))
    run_com_experiment(dataset_dir, 'com_results.csv')
