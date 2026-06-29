import numpy as np
from typing import List, Dict, Any
from pymoo.core.problem import ElementwiseProblem

from core.placement.extreme_point import ExtremePointDecoder
from core.placement.rotation import RotationEngine
from core.metrics.casc import CASCv2, CASCv2Config
from core.metrics.com import CoMModel, CoMConfig

class CargoProblem(ElementwiseProblem):
    def __init__(self, items: List[Dict[str, Any]], container_dims: List[float]):
        self.n_items = len(items)
        super().__init__(
            n_var=2 * self.n_items, 
            n_obj=2,             # f1: CoM Deviation, f2: -CASC
            n_ieq_constr=2,      # g1: Unplaced Volume, g2: Unplaced Count
            xl=np.zeros(2 * self.n_items), # Lower bounds
            xu=np.concatenate([np.ones(self.n_items) * (self.n_items - 1), np.ones(self.n_items) * 6])
        )
        self.items = items
        self.container_dims = container_dims
        
        self.casc_evaluator = CASCv2(CASCv2Config(grid_resolution=10))
        # Realistic config
        self.com_evaluator = CoMModel(CoMConfig(
            target_com=np.array([0.5, 0.5, 1.0]),
            dry_mass=5000.0,
            dry_com=np.array([0.5, 0.5, 1.0])
        ))

    def _evaluate(self, x, out, *args, **kwargs):
        sequence = x[:self.n_items].astype(int)
        orientations = x[self.n_items:].astype(int)
        
        decoder = ExtremePointDecoder(self.container_dims)
        
        unplaced_volume = 0.0
        unplaced_count = 0
        
        for idx in sequence:
            item_data = self.items[idx]
            base_dims = item_data["dimensions_wdh_meters"]
            rot_idx = orientations[idx] - 1
            
            # Get specific rotation
            all_rots = RotationEngine.all_rotations(base_dims)
            # Ensure safe indexing in case mutation throws out of bounds
            rot_idx = max(0, min(rot_idx, len(all_rots)-1))
            rot_dims = all_rots[rot_idx]
            
            packed = decoder.try_pack_item(rot_dims, item_data)
            
            if not packed:
                v = base_dims[0] * base_dims[1] * base_dims[2]
                unplaced_volume += v
                unplaced_count += 1
                
        packed_items = decoder.packed_items
        
        # Constraints (<= 0)
        g1 = unplaced_volume
        g2 = unplaced_count
        
        # Objectives
        # If nothing is packed, we must provide worst-case objective values 
        # so it doesn't accidentally think an empty container is perfect CoM.
        if len(packed_items) == 0:
            f1 = 9999.0 # Terrible CoM
            f2 = 0.0    # Terrible CASC
        else:
            com_res = self.com_evaluator.evaluate_layout(packed_items)
            f1 = com_res['delta_com']
            
            casc_res = self.casc_evaluator.evaluate_layout(packed_items)
            f2 = -casc_res['mean_casc']
        
        out["F"] = [f1, f2]
        out["G"] = [g1, g2]
