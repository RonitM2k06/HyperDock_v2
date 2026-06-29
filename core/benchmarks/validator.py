import json
from typing import Dict, Any

class HyperDockBenchValidator:
    """Validates generated benchmark datasets."""

    @staticmethod
    def validate_schema(data: Dict[str, Any]) -> bool:
        required_keys = ["benchmark_id", "scenario", "scale", "seed", "theoretical_utilization", "containers", "items"]
        for k in required_keys:
            if k not in data:
                print(f"Validation Error: Missing key '{k}'")
                return False
        return True

    @staticmethod
    def validate_masses_and_volumes(data: Dict[str, Any]) -> bool:
        for item in data["items"]:
            if not (0.1 <= item["mass_kg"] <= 80.0):
                print(f"Validation Error: Item {item['item_id']} has invalid mass {item['mass_kg']}")
                return False
            dims = item["dimensions_wdh_meters"]
            if len(dims) != 3 or any(d <= 0 for d in dims):
                print(f"Validation Error: Item {item['item_id']} has invalid dimensions {dims}")
                return False
        return True

    @staticmethod
    def validate_determinism(generator_func, benchmark_id: str, scenario: str, scale: int) -> bool:
        # Run generator twice and assert output is identical
        out1 = generator_func(benchmark_id, scenario, scale)
        out2 = generator_func(benchmark_id, scenario, scale)
        # Convert to json strings to easily compare dicts
        s1 = json.dumps(out1, sort_keys=True)
        s2 = json.dumps(out2, sort_keys=True)
        if s1 != s2:
            print("Validation Error: Determinism failed. Outputs are not identical.")
            return False
        return True

    @staticmethod
    def validate_solvability(data: Dict[str, Any]) -> bool:
        # Solvability validation is a placeholder. 
        # For SC3, theoretically this requires running an exact solver or a massive metaheuristic search.
        # For Phase 1B, we just verify theoretical utilization <= 1.0
        if data["theoretical_utilization"] > 1.0:
            print(f"Validation Error: Theoretical utilization {data['theoretical_utilization']} > 1.0. Unsolvable.")
            return False
        return True

    @classmethod
    def validate_all(cls, data: Dict[str, Any]) -> bool:
        if not cls.validate_schema(data): return False
        if not cls.validate_masses_and_volumes(data): return False
        if not cls.validate_solvability(data): return False
        return True
