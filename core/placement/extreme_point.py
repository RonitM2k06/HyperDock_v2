import numpy as np
from numba import njit, b1, f8, i8
from typing import List, Dict, Any, Tuple

@njit("b1(f8[:], f8[:], f8[:], f8[:])", nopython=True)
def check_overlap_jit(p1, d1, p2, d2):
    x1, y1, z1 = p1[0], p1[1], p1[2]
    w1, d_1, h1 = d1[0], d1[1], d1[2]
    x2, y2, z2 = p2[0], p2[1], p2[2]
    w2, d_2, h2 = d2[0], d2[1], d2[2]

    if (x1 + w1 <= x2 + 1e-5 or x1 >= x2 + w2 - 1e-5 or
        y1 + d_1 <= y2 + 1e-5 or y1 >= y2 + d_2 - 1e-5 or
        z1 + h1 <= z2 + 1e-5 or z1 >= z2 + h2 - 1e-5):
        return False
    return True

@njit("b1(f8[:], f8[:], f8, f8, f8, f8[:,:], f8[:,:], i8)", nopython=True)
def can_pack_jit(point, dimensions, cw, cd, ch, packed_points, packed_dims, num_packed):
    x, y, z = point[0], point[1], point[2]
    w, d, h = dimensions[0], dimensions[1], dimensions[2]

    if x + w > cw + 1e-5 or y + d > cd + 1e-5 or z + h > ch + 1e-5:
        return False

    for i in range(num_packed):
        if check_overlap_jit(point, dimensions, packed_points[i], packed_dims[i]):
            return False
            
    return True

class ExtremePointDecoder:
    def __init__(self, container_dimensions: List[float], max_items: int = 5000):
        self.cw, self.cd, self.ch = container_dimensions
        self.max_items = max_items
        self.reset()

    def reset(self):
        self.packed_items = []
        self.packed_points_arr = np.zeros((self.max_items, 3), dtype=np.float64)
        self.packed_dims_arr = np.zeros((self.max_items, 3), dtype=np.float64)
        self.num_packed = 0
        self.extreme_points = [(0.0, 0.0, 0.0)]
        self.ep_set = set([(0.0, 0.0, 0.0)])

    def can_pack(self, point: Tuple[float, float, float], dimensions: List[float]) -> bool:
        pt = np.array(point, dtype=np.float64)
        dims = np.array(dimensions, dtype=np.float64)
        return can_pack_jit(
            pt, dims, self.cw, self.cd, self.ch, 
            self.packed_points_arr, self.packed_dims_arr, self.num_packed
        )

    def add_item(self, point: Tuple[float, float, float], dimensions: List[float], item_data: Dict[str, Any]):
        self.packed_items.append({
            'point': point,
            'dimensions': dimensions,
            'data': item_data
        })
        
        idx = self.num_packed
        if idx < self.max_items:
            self.packed_points_arr[idx] = point
            self.packed_dims_arr[idx] = dimensions
            self.num_packed += 1
            
        rp = (round(point[0], 5), round(point[1], 5), round(point[2], 5))
        if rp in self.ep_set:
            self.ep_set.remove(rp)
            # Rebuild list from set for iteration
            self.extreme_points = list(self.ep_set)

        x, y, z = point
        w, d, h = dimensions
        
        new_eps = [
            (round(x + w, 5), round(y, 5), round(z, 5)),
            (round(x, 5), round(y + d, 5), round(z, 5)),
            (round(x, 5), round(y, 5), round(z + h, 5))
        ]

        added = False
        for ep in new_eps:
            if ep[0] < self.cw and ep[1] < self.cd and ep[2] < self.ch:
                if ep not in self.ep_set:
                    self.ep_set.add(ep)
                    added = True

        if added or rp not in self.ep_set:
            self.extreme_points = list(self.ep_set)
            self.extreme_points.sort(key=lambda p: (p[2], p[1], p[0]))

    def try_pack_item(self, dimensions: List[float], item_data: Dict[str, Any]) -> bool:
        dims = np.array(dimensions, dtype=np.float64)
        cw, cd, ch = self.cw, self.cd, self.ch
        pts_arr = self.packed_points_arr
        dims_arr = self.packed_dims_arr
        n_packed = self.num_packed
        
        for ep in self.extreme_points:
            pt = np.array(ep, dtype=np.float64)
            if can_pack_jit(pt, dims, cw, cd, ch, pts_arr, dims_arr, n_packed):
                self.add_item(ep, dimensions, item_data)
                return True
        return False
