import sys
import os
import time
import torch
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.metrics.casc import CASCv2, CASCv2Config
from core.metrics.casc_gpu import CASCv2GPU
from core.metrics.com import CoMModel
from core.metrics.com_gpu import CoMModelGPU

def generate_random_layout(n_items=50, container_dims=[10.0, 10.0, 10.0]):
    cw, cd, ch = container_dims
    items = []
    
    for i in range(n_items):
        w = np.random.uniform(0.5, 2.0)
        d = np.random.uniform(0.5, 2.0)
        h = np.random.uniform(0.5, 2.0)
        x = np.random.uniform(0, cw - w)
        y = np.random.uniform(0, cd - d)
        z = np.random.uniform(0, ch - h)
        
        items.append({
            'point': (x, y, z),
            'dimensions': (w, d, h),
            'data': {
                'item_id': f'item_{i}',
                'mass_kg': np.random.uniform(10, 100),
                'category': 'hardware'
            }
        })
    return items

def run_validation():
    print("Testing CPU vs GPU Mathematical Equivalence...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    n_items = 100
    batch_size = 5
    container_dims = [10.0, 10.0, 10.0]
    
    layouts = [generate_random_layout(n_items, container_dims) for _ in range(batch_size)]
    
    # 1. CPU Evaluation
    casc_cpu = CASCv2()
    com_cpu = CoMModel()
    
    cpu_casc_scores = []
    cpu_com_scores = []
    
    t0 = time.time()
    for layout in layouts:
        casc_res = casc_cpu.evaluate_layout(layout)
        cpu_casc_scores.append(casc_res['mean_casc'])
        # CPU CoM relies on 'delta_com' from evaluate_layout
        com_res = com_cpu.evaluate_layout(layout)
        cpu_com_scores.append(com_res['delta_com'])
    cpu_time = time.time() - t0
    
    # 2. Convert to GPU Tensors
    pts_tensor = torch.zeros((batch_size, n_items, 3), dtype=torch.float32)
    dims_tensor = torch.zeros((batch_size, n_items, 3), dtype=torch.float32)
    masses_tensor = torch.zeros((batch_size, n_items), dtype=torch.float32)
    cat_tensor = torch.ones((batch_size, n_items), dtype=torch.float32) # category=hardware (rho=1.0)
    
    for b, layout in enumerate(layouts):
        for i, item in enumerate(layout):
            pts_tensor[b, i] = torch.tensor(item['point'])
            dims_tensor[b, i] = torch.tensor(item['dimensions'])
            masses_tensor[b, i] = item['data']['mass_kg']
            
    pts_tensor = pts_tensor.to(device)
    dims_tensor = dims_tensor.to(device)
    masses_tensor = masses_tensor.to(device)
    cat_tensor = cat_tensor.to(device)
    
    # 3. GPU Evaluation
    casc_gpu = CASCv2GPU(device=device)
    com_gpu = CoMModelGPU(device=device)
    
    t0 = time.time()
    gpu_casc_scores = casc_gpu.evaluate_population(pts_tensor, dims_tensor, masses_tensor, cat_tensor)
    gpu_com_scores = com_gpu.evaluate_population(pts_tensor, dims_tensor, masses_tensor)
    gpu_time = time.time() - t0
    
    gpu_casc_scores = gpu_casc_scores.cpu().numpy()
    gpu_com_scores = gpu_com_scores.cpu().numpy()
    cpu_casc_scores = np.array(cpu_casc_scores)
    cpu_com_scores = np.array(cpu_com_scores)
    
    casc_diff = np.abs(cpu_casc_scores - gpu_casc_scores).max()
    com_diff = np.abs(cpu_com_scores - gpu_com_scores).max()
    
    print(f"\nResults (Batch Size: {batch_size}, Items: {n_items}):")
    print(f"Max CASC Difference: {casc_diff:.6e}")
    print(f"Max CoM Difference:  {com_diff:.6e}")
    
    print(f"\nPerformance:")
    print(f"CPU Time: {cpu_time:.4f}s")
    print(f"GPU Time: {gpu_time:.4f}s (Speedup: {cpu_time/max(gpu_time, 1e-5):.2f}x)")
    
    if casc_diff < 1e-4 and com_diff < 1e-4:
        print("\n✅ VALIDATION PASSED: CPU and GPU outputs are mathematically equivalent.")
    else:
        print("\n❌ VALIDATION FAILED: High variance between CPU and GPU outputs.")
        print(f"CPU CASC: {cpu_casc_scores}")
        print(f"GPU CASC: {gpu_casc_scores}")
        print(f"CPU CoM:  {cpu_com_scores}")
        print(f"GPU CoM:  {gpu_com_scores}")

if __name__ == '__main__':
    run_validation()
