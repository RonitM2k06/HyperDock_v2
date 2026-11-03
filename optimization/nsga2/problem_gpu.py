import numpy as np
import torch
import multiprocessing as mp
from typing import List, Dict, Any
from pymoo.core.problem import Problem

from core.placement.extreme_point import ExtremePointDecoder
from core.placement.rotation import RotationEngine
from core.metrics.casc import CASCv2Config
from core.metrics.casc_gpu import CASCv2GPU
from core.metrics.com import CoMConfig
from core.metrics.com_gpu import CoMModelGPU

def pack_chromosome(args):
    """Worker function for parallel CPU packing."""
    idx, sequence, orientations, items, container_dims = args
    decoder = ExtremePointDecoder(container_dims)
    
    unplaced_volume = 0.0
    unplaced_count = 0
    n_items = len(items)
    
    for i in sequence:
        item_data = items[i]
        base_dims = item_data["dimensions_wdh_meters"]
        rot_idx = orientations[i] - 1
        
        all_rots = RotationEngine.all_rotations(base_dims)
        rot_idx = max(0, min(rot_idx, len(all_rots)-1))
        rot_dims = all_rots[rot_idx]
        
        packed = decoder.try_pack_item(rot_dims, item_data)
        
        if not packed:
            v = base_dims[0] * base_dims[1] * base_dims[2]
            unplaced_volume += v
            unplaced_count += 1
            
    packed_items = decoder.packed_items
    
    # Extract tensor data for this layout
    pts = np.zeros((n_items, 3), dtype=np.float32)
    dims = np.zeros((n_items, 3), dtype=np.float32)
    masses = np.zeros(n_items, dtype=np.float32)
    categories = np.zeros(n_items, dtype=np.float32)
    
    for j, p_item in enumerate(packed_items):
        pts[j] = p_item['point']
        dims[j] = p_item['dimensions']
        masses[j] = p_item['data'].get('mass_kg', 0.0)
        
        cat = p_item['data'].get('category', 'hardware')
        rho = 0.4 if cat in ['consumable', 'waste'] else 1.0
        categories[j] = rho
        
    num_packed = len(packed_items)
    return idx, unplaced_volume, unplaced_count, num_packed, pts, dims, masses, categories

class CargoProblemGPU(Problem):
    def __init__(self, items: List[Dict[str, Any]], container_dims: List[float], n_workers: int = 4):
        self.n_items = len(items)
        super().__init__(
            n_var=2 * self.n_items, 
            n_obj=2,             # f1: CoM Deviation, f2: -CASC
            n_ieq_constr=2,      # g1: Unplaced Volume, g2: Unplaced Count
            xl=np.zeros(2 * self.n_items),
            xu=np.concatenate([np.ones(self.n_items) * (self.n_items - 1), np.ones(self.n_items) * 6])
        )
        self.items = items
        self.container_dims = container_dims
        self.n_workers = n_workers
        
        # PyTorch Evaluators
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.casc_gpu = CASCv2GPU(CASCv2Config(grid_resolution=10), device=self.device)
        self.com_gpu = CoMModelGPU(CoMConfig(
            target_com=np.array([0.5, 0.5, 1.0]),
            dry_mass=5000.0,
            dry_com=np.array([0.5, 0.5, 1.0])
        ), device=self.device)
        
        if self.n_workers > 1:
            self.pool = mp.Pool(processes=self.n_workers)
        else:
            self.pool = None

    def __del__(self):
        if hasattr(self, 'pool') and self.pool is not None:
            self.pool.close()
            self.pool.join()

    def _evaluate(self, X, out, *args, **kwargs):
        pop_size = X.shape[0]
        
        # 1. Parallel CPU Packing
        tasks = []
        for i in range(pop_size):
            sequence = X[i, :self.n_items].astype(int)
            orientations = X[i, self.n_items:].astype(int)
            tasks.append((i, sequence, orientations, self.items, self.container_dims))
            
        if self.pool is not None:
            results = self.pool.map(pack_chromosome, tasks)
        else:
            results = list(map(pack_chromosome, tasks))
        
        # 2. Assemble Batch Tensors
        g1 = np.zeros(pop_size)
        g2 = np.zeros(pop_size)
        f1 = np.zeros(pop_size)
        f2 = np.zeros(pop_size)
        
        all_pts = np.zeros((pop_size, self.n_items, 3), dtype=np.float32)
        all_dims = np.zeros((pop_size, self.n_items, 3), dtype=np.float32)
        all_masses = np.zeros((pop_size, self.n_items), dtype=np.float32)
        all_categories = np.zeros((pop_size, self.n_items), dtype=np.float32)
        all_valid = np.zeros((pop_size, self.n_items), dtype=bool)  # True for packed items only

        valid_mask = np.zeros(pop_size, dtype=bool)

        for res in results:
            idx, unplaced_vol, unplaced_count, num_packed, pts, dims, masses, cats = res
            g1[idx] = unplaced_vol
            g2[idx] = unplaced_count

            if num_packed == 0:
                f1[idx] = 9999.0
                f2[idx] = 0.0
            else:
                all_pts[idx] = pts
                all_dims[idx] = dims
                all_masses[idx] = masses
                all_categories[idx] = cats
                all_valid[idx, :num_packed] = True  # pack_chromosome writes packed items first
                valid_mask[idx] = True

        # 3. Batched GPU Evaluation
        if np.any(valid_mask):
            pts_tensor = torch.from_numpy(all_pts[valid_mask]).to(self.device)
            dims_tensor = torch.from_numpy(all_dims[valid_mask]).to(self.device)
            masses_tensor = torch.from_numpy(all_masses[valid_mask]).to(self.device)
            cat_tensor = torch.from_numpy(all_categories[valid_mask]).to(self.device)
            valid_tensor = torch.from_numpy(all_valid[valid_mask]).to(self.device)

            with torch.no_grad():
                casc_scores = self.casc_gpu.evaluate_population(pts_tensor, dims_tensor, masses_tensor, cat_tensor, valid_tensor)
                com_scores = self.com_gpu.evaluate_population(pts_tensor, dims_tensor, masses_tensor)
                
            casc_scores_np = casc_scores.cpu().numpy()
            com_scores_np = com_scores.cpu().numpy()
            
            f1[valid_mask] = com_scores_np
            f2[valid_mask] = -casc_scores_np
            
        out["F"] = np.column_stack([f1, f2])
        out["G"] = np.column_stack([g1, g2])
