import numpy as np
from numba import njit, b1, f8, i8
from typing import List, Dict, Any, Set
from dataclasses import dataclass

@dataclass
class CASCv2Config:
    L_arm: float = 0.75
    gamma: float = 2.0
    lambda_occ: float = 1.0
    C_grasp: float = 2.0
    kappa: float = 0.05
    grid_resolution: int = 20

@njit("b1(f8, f8, f8, f8, f8, f8, f8, f8, f8, f8, f8, f8)", nopython=True)
def is_blocking_jit(ix, iy, iz, iw, id_y, ih, jx, jy, jz, jw, jd_y, jh):
    if jz >= iz - 1e-5:
        return False
    if (ix + iw <= jx + 1e-5 or ix >= jx + jw - 1e-5 or
        iy + id_y <= jy + 1e-5 or iy >= jy + jd_y - 1e-5):
        return False
    return True

@njit("b1[:,:](f8[:,:], f8[:,:], i8)", nopython=True)
def build_blocking_graph_jit(points, dims, n):
    graph = np.zeros((n, n), dtype=np.bool_)
    for i in range(n):
        for j in range(n):
            if i != j:
                ix, iy, iz = points[i]
                iw, id_y, ih = dims[i]
                jx, jy, jz = points[j]
                jw, jd_y, jh = dims[j]
                if is_blocking_jit(ix, iy, iz, iw, id_y, ih, jx, jy, jz, jw, jd_y, jh):
                    graph[i, j] = True
    return graph

@njit("b1[:](i8, b1[:,:], i8)", nopython=True)
def get_transitive_blockers_jit(target_idx, graph, n):
    visited = np.zeros(n, dtype=np.bool_)
    queue = np.zeros(n, dtype=np.int64)
    head = 0
    tail = 0
    
    queue[tail] = target_idx
    tail += 1
    
    while head < tail:
        curr = queue[head]
        head += 1
        for blocker in range(n):
            if graph[curr, blocker] and not visited[blocker]:
                visited[blocker] = True
                queue[tail] = blocker
                tail += 1
    return visited

@njit("f8(i8, f8[:,:], f8[:,:], i8, i8, f8)", nopython=True)
def compute_occlusion_penalty_jit(target_idx, points, dims, n, grid_res, lambda_occ):
    tx, ty, tz = points[target_idx]
    tw, td_y, th = dims[target_idx]
    
    blockers_idx = np.zeros(n, dtype=np.int64)
    num_blockers = 0
    for j in range(n):
        if j != target_idx and points[j, 2] < tz - 1e-5:
            blockers_idx[num_blockers] = j
            num_blockers += 1
            
    if num_blockers == 0:
        return 0.0
        
    xs = np.linspace(tx + 1e-5, tx + tw - 1e-5, grid_res)
    ys = np.linspace(ty + 1e-5, ty + td_y - 1e-5, grid_res)
    
    total_points = grid_res * grid_res
    visible_points = total_points
    
    for x in xs:
        for y in ys:
            is_occluded = False
            for k in range(num_blockers):
                j = blockers_idx[k]
                jx, jy, _ = points[j]
                jw, jd_y, _ = dims[j]
                if jx <= x <= jx + jw and jy <= y <= jy + jd_y:
                    is_occluded = True
                    break
            if is_occluded:
                visible_points -= 1
                
    A_vis_ratio = float(visible_points) / float(total_points)
    return lambda_occ * (1.0 - A_vis_ratio)

class CASCv2:
    def __init__(self, config: CASCv2Config = None):
        self.config = config or CASCv2Config()

    def _get_rigidity(self, category: str) -> float:
        if category in ['consumable', 'waste']:
            return 0.4
        return 1.0

    def evaluate_layout(self, packed_items: List[Dict]) -> Dict[str, Any]:
        n = len(packed_items)
        if n == 0:
            return {'mean_casc': 0.0, 'total_casc': 0.0, 'item_details': []}

        points = np.zeros((n, 3), dtype=np.float64)
        dims = np.zeros((n, 3), dtype=np.float64)
        
        for i, item in enumerate(packed_items):
            points[i] = item['point']
            dims[i] = item['dimensions']
            
        graph = build_blocking_graph_jit(points, dims, n)
        
        results = []
        total_casc = 0.0
        
        for i, item in enumerate(packed_items):
            z_i = points[i, 2]
            c_reach = (z_i / self.config.L_arm) ** self.config.gamma
            
            c_occ = compute_occlusion_penalty_jit(
                i, points, dims, n, self.config.grid_resolution, self.config.lambda_occ
            )
            
            transitive_mask = get_transitive_blockers_jit(i, graph, n)
            c_ext = 0.0
            for b_idx in range(n):
                if transitive_mask[b_idx]:
                    b_item = packed_items[b_idx]['data']
                    rho = self._get_rigidity(b_item.get('category', 'hardware'))
                    m_j = b_item.get('mass_kg', 0.0)
                    c_ext += rho * (self.config.C_grasp + self.config.kappa * m_j)
                    
            c_total = c_reach + c_occ + c_ext
            casc_score = float(np.exp(-c_total))
            total_casc += casc_score
            
            item_id = item.get('data', {}).get('item_id', f'unknown_{i}')
            
            results.append({
                'item_id': item_id,
                'casc_score': casc_score,
                'c_reach': c_reach,
                'c_occ': c_occ,
                'c_ext': c_ext,
                'transitive_blockers_count': int(np.sum(transitive_mask))
            })
            
        return {
            'mean_casc': total_casc / n,
            'total_casc': total_casc,
            'item_details': results
        }
