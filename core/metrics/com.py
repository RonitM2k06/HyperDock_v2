import numpy as np
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass, field

@dataclass
class ContainerTransform:
    """Transformation matrices and offsets for a specific container in the SCS."""
    container_id: str
    R_M: np.ndarray = field(default_factory=lambda: np.zeros(3))
    Lambda_M: np.ndarray = field(default_factory=lambda: np.eye(3))
    R_c: np.ndarray = field(default_factory=lambda: np.zeros(3))
    Lambda_c: np.ndarray = field(default_factory=lambda: np.eye(3))

@dataclass
class CoMConfig:
    """Externalized parameters for CoM Optimization."""
    target_com: np.ndarray = field(default_factory=lambda: np.zeros(3))
    dry_mass: float = 5000.0
    dry_com: np.ndarray = field(default_factory=lambda: np.zeros(3))
    # Tolerances for ellipsoidal objective
    epsilon_x: float = 0.03 # 3cm
    epsilon_y: float = 0.10 # 10cm
    epsilon_z: float = 0.10 # 10cm

class CoMModel:
    """
    Center of Mass (CoM) Optimization Model.
    Computes absolute mass properties in the Spacecraft Coordinate System (SCS).
    """
    def __init__(self, config: CoMConfig = None, container_transforms: Dict[str, ContainerTransform] = None):
        self.config = config or CoMConfig()
        self.transforms = container_transforms or {}

    def _get_transform(self, container_id: str) -> ContainerTransform:
        """Returns the transform for a container, defaulting to origin if not specified."""
        if container_id not in self.transforms:
            return ContainerTransform(container_id=container_id)
        return self.transforms[container_id]

    def _compute_absolute_item_com(self, packed_item: Dict[str, Any], container_id: str) -> np.ndarray:
        """
        Maps a packed item's local coordinates to the absolute SCS coordinates.
        r_i = R_M + Lambda_M * (R_c + Lambda_c * (p_i + r_i_local_com))
        """
        transform = self._get_transform(container_id)
        
        p_i = np.array(packed_item['point'])
        w, d, h = packed_item['dimensions']
        r_i_local_com = np.array([w / 2.0, d / 2.0, h / 2.0])
        
        local_center = p_i + r_i_local_com
        
        # Apply kinematic chain
        c_rot = transform.Lambda_c @ local_center
        c_pos = transform.R_c + c_rot
        m_rot = transform.Lambda_M @ c_pos
        r_i_abs = transform.R_M + m_rot
        
        return r_i_abs

    def evaluate_layout(self, packed_items: List[Dict], container_id: str = "default") -> Dict[str, Any]:
        """
        Evaluates the CoM metrics for a given layout of items.
        Assumes all items in the list belong to container_id unless explicitly defined in item['data'].
        """
        total_mass = self.config.dry_mass
        mass_moment = self.config.dry_mass * self.config.dry_com
        
        item_details = []
        
        for item in packed_items:
            cid = item.get('data', {}).get('container_id', container_id)
            mass = float(item.get('data', {}).get('mass_kg', 0.0))
            
            if mass <= 0.0:
                continue
                
            r_i = self._compute_absolute_item_com(item, cid)
            
            total_mass += mass
            mass_moment += mass * r_i
            
            item_details.append({
                'item_id': item.get('data', {}).get('item_id', 'unknown'),
                'mass_kg': mass,
                'abs_com': r_i.tolist()
            })
            
        if total_mass > 0:
            system_com = mass_moment / total_mass
        else:
            system_com = np.zeros(3)
            
        # Euclidean Deviation
        delta_com = np.linalg.norm(system_com - self.config.target_com)
        
        # Ellipsoidal Deviation
        diff = system_com - self.config.target_com
        ellipsoidal_dev = np.sqrt(
            (diff[0] / self.config.epsilon_x)**2 +
            (diff[1] / self.config.epsilon_y)**2 +
            (diff[2] / self.config.epsilon_z)**2
        )
        
        return {
            'system_com': system_com.tolist(),
            'total_mass': total_mass,
            'delta_com': float(delta_com),
            'ellipsoidal_dev': float(ellipsoidal_dev),
            'item_details': item_details
        }
