import torch
import numpy as np
from typing import List, Dict, Any

class CASCv2GPU:
    def __init__(self, config=None, device=None):
        from .casc import CASCv2Config
        self.config = config or CASCv2Config()
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def evaluate_population(self, pop_points: torch.Tensor, pop_dims: torch.Tensor, pop_masses: torch.Tensor, pop_categories: torch.Tensor, pop_valid: torch.Tensor = None) -> torch.Tensor:
        """
        Evaluates CASC for an entire NSGA-II population simultaneously.

        Args:
            pop_points: [batch_size, num_items, 3] tensor of item coordinates.
            pop_dims: [batch_size, num_items, 3] tensor of item dimensions.
            pop_masses: [batch_size, num_items] tensor of item masses.
            pop_categories: [batch_size, num_items] tensor of rigidity coefficients (rho).
            pop_valid: optional [batch_size, num_items] bool tensor. True marks a packed
                item. Rows that are False are padding for unplaced items: they are
                excluded from the blocking graph and from the mean. Without this mask,
                padding rows act as real items at the origin and bias the score
                (see tests/test_casc_gpu_mask.py).

        Returns:
            total_casc: [batch_size] tensor of mean CASC scores over packed items per layout.
        """
        B, N, _ = pop_points.shape

        if N == 0:
            return torch.zeros(B, device=self.device)

        # 1. Reach Penalty
        z = pop_points[:, :, 2] # [B, N]
        c_reach = (z / self.config.L_arm) ** self.config.gamma

        # 2. Build Blocking Graph
        # A blocker j must be in front of i (z_j < z_i)
        z_mask = pop_points[:, None, :, 2] < (pop_points[:, :, None, 2] - 1e-5) # [B, N, N]
        
        # AABB Overlap in XY plane
        ix = pop_points[:, :, None, 0]
        iy = pop_points[:, :, None, 1]
        iw = pop_dims[:, :, None, 0]
        id_y = pop_dims[:, :, None, 1]
        
        jx = pop_points[:, None, :, 0]
        jy = pop_points[:, None, :, 1]
        jw = pop_dims[:, None, :, 0]
        jd_y = pop_dims[:, None, :, 1]
        
        # ix + iw <= jx or ix >= jx + jw
        overlap_x = ~( (ix + iw <= jx + 1e-5) | (ix >= jx + jw - 1e-5) )
        overlap_y = ~( (iy + id_y <= jy + 1e-5) | (iy >= jy + jd_y - 1e-5) )
        
        blocking_graph = z_mask & overlap_x & overlap_y # [B, N, N] (i blocked by j)
        if pop_valid is not None:
            # Padding rows neither block nor are blocked.
            valid_pair = pop_valid[:, :, None] & pop_valid[:, None, :]
            blocking_graph = blocking_graph & valid_pair

        # 3. Transitive Blockers (Matrix Power / Warshall approximation for DAG)
        transitive_graph = blocking_graph.clone()
        for _ in range(N):
            # If i is blocked by k, and k is blocked by j, then i is blocked by j
            new_graph = transitive_graph | torch.bmm(transitive_graph.float(), blocking_graph.float()).bool()
            if torch.equal(new_graph, transitive_graph):
                break
            transitive_graph = new_graph
            
        # 4. Extraction Penalty
        extraction_cost = pop_categories * (self.config.C_grasp + self.config.kappa * pop_masses) # [B, N]
        # Sum extraction costs of all transitive blockers
        c_ext = torch.bmm(transitive_graph.float(), extraction_cost.unsqueeze(2)).squeeze(2) # [B, N]
        
        # 5. Occlusion Penalty (Batched grid raycasting)
        res = self.config.grid_resolution
        c_occ = torch.zeros(B, N, device=self.device)
        
        # Create normalized grid [0, 1] x [0, 1]
        grid_norm = torch.linspace(1e-5, 1.0 - 1e-5, res, device=self.device)
        gy, gx = torch.meshgrid(grid_norm, grid_norm, indexing='ij') # [res, res]
        grid_flat = torch.stack([gx.flatten(), gy.flatten()], dim=-1) # [res^2, 2]
        
        # To prevent massive VRAM OOM (B*N*N*res^2 = 2.5GB per operation for N=500, B=100)
        # We iterate over the batch dimension. A python loop of 100 is trivial overhead.
        for b in range(B):
            b_ix = ix[b] # [N, 1]
            b_iy = iy[b]
            b_iw = iw[b]
            b_id_y = id_y[b]
            
            b_jx = jx[b] # [1, N]
            b_jy = jy[b]
            b_jw = jw[b]
            b_jd_y = jd_y[b]
            
            target_grid_x = b_ix + b_iw * grid_flat[:, 0].view(1, -1) # [N, res^2]
            target_grid_y = b_iy + b_id_y * grid_flat[:, 1].view(1, -1) # [N, res^2]
            
            tx = target_grid_x.unsqueeze(1) # [N, 1, res^2]
            ty = target_grid_y.unsqueeze(1) # [N, 1, res^2]
            
            cbjx = b_jx.unsqueeze(2) # [1, N, 1]
            cbjw = b_jw.unsqueeze(2) # [1, N, 1]
            cbjy = b_jy.unsqueeze(2) # [1, N, 1]
            cbjd_y = b_jd_y.unsqueeze(2) # [1, N, 1]
            
            # [N, N, res^2] max memory = 500*500*100 = 25MB (very safe for VRAM)
            is_hit = (tx >= cbjx) & (tx <= cbjx + cbjw) & (ty >= cbjy) & (ty <= cbjy + cbjd_y)
            valid_hit = is_hit & blocking_graph[b].unsqueeze(2)
            
            point_occluded = valid_hit.any(dim=1) # [N, res^2]
            visible_points = res**2 - point_occluded.sum(dim=1).float() # [N]
            A_vis_ratio = visible_points / (res**2)
            c_occ[b] = self.config.lambda_occ * (1.0 - A_vis_ratio)
        
        # Total CASC mapping
        c_total = c_reach + c_occ + c_ext # [B, N]
        casc_scores = torch.exp(-c_total) # [B, N]
        
        if pop_valid is None:
            return casc_scores.mean(dim=1) # [B] (returning mean CASC as the objective)
        v = pop_valid.to(casc_scores.dtype)
        return (casc_scores * v).sum(dim=1) / v.sum(dim=1).clamp_min(1.0) # [B] mean over packed items

