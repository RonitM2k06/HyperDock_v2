from typing import List, Dict, Any, Tuple
from core.placement.extreme_point import ExtremePointDecoder
from core.placement.rotation import RotationEngine

class BaselinePacker:
    """Base class for heuristic baseline packers."""
    
    def __init__(self, container_dimensions: List[float]):
        self.decoder = ExtremePointDecoder(container_dimensions)

    def _pack_sequence(self, items: List[Dict[str, Any]]) -> Tuple[int, float]:
        """
        Attempts to pack a given sequence of items using First-Fit Extreme Point strategy.
        Tries all 6 rotations at each extreme point.
        Returns (number of packed items, total packed volume).
        """
        self.decoder.reset()
        packed_count = 0
        packed_volume = 0.0

        for item in items:
            base_dims = item["dimensions_wdh_meters"]
            packed = False
            
            # Try all 6 rotations
            rotations = RotationEngine.all_rotations(base_dims)
            
            # Try to pack at each extreme point (Extreme points are already sorted by z, y, x)
            for ep in self.decoder.extreme_points:
                for rot_idx, rot_dims in enumerate(rotations):
                    if self.decoder.can_pack(ep, rot_dims):
                        # Pack it
                        item_data_copy = item.copy()
                        item_data_copy['rotation_index'] = rot_idx + 1
                        self.decoder.add_item(ep, rot_dims, item_data_copy)
                        packed_count += 1
                        packed_volume += (rot_dims[0] * rot_dims[1] * rot_dims[2])
                        packed = True
                        break
                if packed:
                    break

        return packed_count, packed_volume


class FirstFitPacker(BaselinePacker):
    """Packs items exactly in the sequence they appear."""
    
    def pack(self, items: List[Dict[str, Any]]) -> Tuple[int, float]:
        return self._pack_sequence(items)

class BestFitDecreasingVolumePacker(BaselinePacker):
    """Sorts items by volume (descending) before packing."""
    
    def pack(self, items: List[Dict[str, Any]]) -> Tuple[int, float]:
        sorted_items = sorted(
            items, 
            key=lambda i: i["dimensions_wdh_meters"][0] * i["dimensions_wdh_meters"][1] * i["dimensions_wdh_meters"][2], 
            reverse=True
        )
        return self._pack_sequence(sorted_items)

class BestFitDecreasingPriorityPacker(BaselinePacker):
    """Sorts items by priority (descending) before packing."""
    
    def pack(self, items: List[Dict[str, Any]]) -> Tuple[int, float]:
        sorted_items = sorted(
            items, 
            key=lambda i: i["priority"], 
            reverse=True
        )
        return self._pack_sequence(sorted_items)
