from typing import List

class RotationEngine:
    """Provides orthogonal rotation transformations for 3D items."""

    @staticmethod
    def get_rotation(dimensions: List[float], rotation_index: int) -> List[float]:
        """
        Returns the dimensions of the item given a rotation index [1-6].
        1: (w, d, h)
        2: (w, h, d)
        3: (d, w, h)
        4: (d, h, w)
        5: (h, w, d)
        6: (h, d, w)
        """
        w, d, h = dimensions
        if rotation_index == 1: return [w, d, h]
        elif rotation_index == 2: return [w, h, d]
        elif rotation_index == 3: return [d, w, h]
        elif rotation_index == 4: return [d, h, w]
        elif rotation_index == 5: return [h, w, d]
        elif rotation_index == 6: return [h, d, w]
        else:
            raise ValueError(f"Invalid rotation index {rotation_index}. Must be 1-6.")

    @staticmethod
    def all_rotations(dimensions: List[float]) -> List[List[float]]:
        """Returns all 6 possible rotations."""
        return [RotationEngine.get_rotation(dimensions, i) for i in range(1, 7)]
