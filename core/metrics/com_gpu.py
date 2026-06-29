import torch
from typing import List

class CoMModelGPU:
    """
    Center of Mass Evaluation Model.
    Batched GPU implementation using PyTorch tensors.
    """
    def __init__(self, config=None, device=None):
        from .com import CoMConfig
        self.config = config or CoMConfig()
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def evaluate_population(self, pop_points: torch.Tensor, pop_dims: torch.Tensor, pop_masses: torch.Tensor) -> torch.Tensor:
        """
        Evaluates CoM deviation for an entire NSGA-II population simultaneously.
        
        Args:
            pop_points: [batch_size, num_items, 3] tensor of item coordinates.
            pop_dims: [batch_size, num_items, 3] tensor of item dimensions.
            pop_masses: [batch_size, num_items] tensor of item masses.
            
        Returns:
            com_deviation: [batch_size] tensor of CoM Euclidean deviations.
        """
        B, N, _ = pop_points.shape
        
        target_com = torch.tensor(self.config.target_com, device=self.device, dtype=torch.float32)
        dry_mass = self.config.dry_mass
        dry_com = torch.tensor(self.config.dry_com, device=self.device, dtype=torch.float32)
        
        if N == 0:
            sys_com = dry_com
            return torch.norm(sys_com - target_com).unsqueeze(0).repeat(B)
            
        # Absolute CoM for each item is the center of its AABB
        # Assuming no kinematic rotation/offset for this simplified model (R_M=0, Lambda_M=I)
        item_coms = pop_points + (pop_dims / 2.0) # [B, N, 3]
        
        # Weighted sum: sum(mass * com) / sum(mass)
        masses = pop_masses.unsqueeze(2) # [B, N, 1]
        
        item_mass_moment = (item_coms * masses).sum(dim=1) # [B, 3]
        total_item_mass = pop_masses.sum(dim=1) # [B]
        
        system_mass = dry_mass + total_item_mass # [B]
        system_mass_moment = (dry_mass * dry_com) + item_mass_moment # [B, 3]
        
        system_coms = system_mass_moment / system_mass.unsqueeze(1) # [B, 3]
        
        # Euclidean Deviation
        deviations = torch.norm(system_coms - target_com, dim=1) # [B]
        
        return deviations
