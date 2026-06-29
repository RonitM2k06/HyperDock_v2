import json
import hashlib
import numpy as np
from typing import Dict, Any, List

class HyperDockBenchGenerator:
    """Generates synthetic dataset instances for HyperDock-Bench-V1."""

    CTB_DIMENSIONS = {
        "SINGLE": [0.502, 0.425, 0.248],
        "HALF": [0.502, 0.425, 0.124],
        "QUARTER": [0.251, 0.425, 0.124],
        "DOUBLE": [0.502, 0.425, 0.496]
    }

    def __init__(self, benchmark_id: str, scenario: str, scale: int):
        self.benchmark_id = benchmark_id
        self.scenario = scenario
        self.scale = scale
        self.seed = self._get_deterministic_seed(benchmark_id)
        self.rng = np.random.default_rng(self.seed)

    def _get_deterministic_seed(self, benchmark_id: str) -> int:
        return int(hashlib.md5(benchmark_id.encode('utf-8')).hexdigest(), 16) % (2**32)

    def _generate_dimensions(self) -> List[float]:
        choices = ["SINGLE", "HALF", "QUARTER", "DOUBLE", "IRREGULAR"]
        probs = [0.4, 0.3, 0.1, 0.1, 0.1]
        choice = self.rng.choice(choices, p=probs)
        if choice == "IRREGULAR":
            return [round(self.rng.uniform(0.1, 0.8), 3) for _ in range(3)]
        return self.CTB_DIMENSIONS[choice].copy()

    def _generate_item(self, idx: int) -> Dict[str, Any]:
        if self.scenario == "SC1":
            cat_probs = [0.5, 0.3, 0.1, 0.1] # Consumables, Hardware, Payload, Safety
        elif self.scenario == "SC2":
            cat_probs = [0.2, 0.3, 0.1, 0.4] # Safety skewed
        else: # SC3
            cat_probs = [0.5, 0.3, 0.1, 0.1]

        categories = ["consumable", "hardware", "payload", "safety"]
        cat = self.rng.choice(categories, p=cat_probs)

        if cat == "consumable":
            mass = self.rng.lognormal(1.5, 0.6)
            crit = "essential"
            lam = 4.0
        elif cat == "hardware":
            mass = self.rng.lognormal(2.2, 0.8)
            crit = "nominal"
            lam = 0.5
        elif cat == "safety":
            mass = self.rng.lognormal(1.0, 0.4)
            crit = "medical_emergency"
            lam = 0.01
        else:
            mass = self.rng.lognormal(1.8, 0.5)
            crit = "nominal"
            lam = 0.2

        mass = float(np.clip(mass, 0.1, 80.0))
        dims = self._generate_dimensions()

        return {
            "item_id": f"ITM-{idx:04d}",
            "category": cat,
            "criticality": crit,
            "preferred_zone": "GENERIC",
            "dimensions_wdh_meters": dims,
            "mass_kg": round(mass, 2),
            "priority": int(self.rng.integers(1, 101)),
            "retrieval_lambda": lam
        }

    def generate(self) -> Dict[str, Any]:
        items = [self._generate_item(i) for i in range(1, self.scale + 1)]
        
        # Determine total volume of items
        total_vol = sum([d[0]*d[1]*d[2] for d in [item["dimensions_wdh_meters"] for item in items]])
        
        # For SC3, container volume is ~ 1.04 * total_vol (96% density limit)
        # For SC1, SC2, container volume is ~ 1.25 * total_vol (80% density limit)
        vol_multiplier = 1.04 if self.scenario == "SC3" else 1.25
        target_container_vol = total_vol * vol_multiplier

        # Enforce exact theoretical density by creating ONE perfectly sized container
        # Ensure dimensions are large enough for the max item size (0.8m)
        w = max(1.0, (target_container_vol) ** (1/3))
        d = max(1.0, (target_container_vol) ** (1/3))
        h = target_container_vol / (w * d)
        
        containers = [
            {
                "container_id": "C-HAB-01",
                "zone": "HABITATION",
                "absolute_origin_xyz_meters": [0.0, 0.0, 0.0],
                "dimensions_wdh_meters": [round(w, 4), round(d, 4), round(h, 4)],
                "open_face_normal": [0, -1, 0]
            }
        ]
        
        # Calculate actual container volume based on rounded dimensions
        actual_container_vol = round(w, 4) * round(d, 4) * round(h, 4)

        return {
            "benchmark_id": self.benchmark_id,
            "scenario": self.scenario,
            "scale": self.scale,
            "seed": self.seed,
            "theoretical_utilization": round(total_vol / actual_container_vol, 4),
            "containers": containers,
            "items": items
        }

if __name__ == "__main__":
    gen = HyperDockBenchGenerator("HB-V1-SC1-100", "SC1", 100)
    data = gen.generate()
    print(f"Generated {data['benchmark_id']} with {len(data['items'])} items and {len(data['containers'])} containers.")
    print(f"Theoretical utilization: {data['theoretical_utilization']}")
