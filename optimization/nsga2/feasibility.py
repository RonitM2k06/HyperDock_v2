"""
Feasibility-aware evaluation for the cargo packing problem.

Method: "Baldwinian repair with a fallback decode order".

For a chromosome (sequence, orientations):
  1. Decode in the given sequence order (identical to CargoProblem).
  2. If that decode leaves any item unplaced (G > 0), decode a second time with
     the same orientations but the sequence re-sorted by descending item volume
     (the decreasing-volume rule used by the BFD baselines).
  3. Keep whichever decode leaves less unplaced volume (ties keep step 1).
  4. Objectives and constraints are computed from the kept decode.

The genotype is NOT modified (Baldwinian: the repair influences fitness only,
the chromosome stays as the search produced it). Evaluation cost is therefore
at most two decodes per chromosome; the matched-budget runner counts one
evaluation per chromosome and reports decoder calls separately.

This module does not change CargoProblem, so the baseline NSGA-II behaviour is
unchanged and remains the comparison point.
"""
import time
import numpy as np
from pymoo.core.problem import ElementwiseProblem

from core.placement.extreme_point import ExtremePointDecoder
from core.placement.rotation import RotationEngine
from optimization.nsga2.problem import CargoProblem


def _volume(dims):
    return float(dims[0]) * float(dims[1]) * float(dims[2])


def decode_layout(items, container_dims, sequence, orientations):
    """
    Decode one chromosome with the Extreme Point decoder.

    Returns (unplaced_volume, unplaced_count, packed_items, decoder_calls=1).
    Semantics match CargoProblem._evaluate exactly.
    """
    decoder = ExtremePointDecoder(container_dims)
    unplaced_volume = 0.0
    unplaced_count = 0
    for idx in sequence:
        item_data = items[idx]
        base_dims = item_data["dimensions_wdh_meters"]
        rot_idx = int(orientations[idx]) - 1
        all_rots = RotationEngine.all_rotations(base_dims)
        rot_idx = max(0, min(rot_idx, len(all_rots) - 1))
        rot_dims = all_rots[rot_idx]
        if not decoder.try_pack_item(rot_dims, item_data):
            unplaced_volume += _volume(base_dims)
            unplaced_count += 1
    return unplaced_volume, unplaced_count, decoder.packed_items


def descending_volume_order(items, sequence):
    """Re-sort a sequence by descending item volume (ties keep original order)."""
    seq = [int(i) for i in sequence]
    return sorted(seq, key=lambda i: -_volume(items[i]["dimensions_wdh_meters"]))


class CargoProblemRepair(CargoProblem):
    """
    CargoProblem with the feasibility-aware fallback decode described above.

    Same objectives F = [CoM deviation, -mean CASC] and constraints
    G = [unplaced volume, unplaced count] as CargoProblem.
    """

    def __init__(self, items, container_dims):
        super().__init__(items=items, container_dims=container_dims)
        self.decodes_total = 0          # decoder calls across all evaluations
        self.repaired_count = 0         # evaluations where the fallback decode was kept
        self.eval_time_s = 0.0

    def _objectives(self, packed_items):
        if len(packed_items) == 0:
            return 9999.0, 0.0
        com_res = self.com_evaluator.evaluate_layout(packed_items)
        casc_res = self.casc_evaluator.evaluate_layout(packed_items)
        return float(com_res["delta_com"]), -float(casc_res["mean_casc"])

    def _evaluate(self, x, out, *args, **kwargs):
        t0 = time.perf_counter()
        sequence = x[: self.n_items].astype(int)
        orientations = x[self.n_items:].astype(int)

        uv, uc, packed = decode_layout(self.items, self.container_dims, sequence, orientations)
        decodes = 1

        if uc > 0:
            seq2 = descending_volume_order(self.items, sequence)
            uv2, uc2, packed2 = decode_layout(self.items, self.container_dims, seq2, orientations)
            decodes += 1
            if uv2 < uv:
                uv, uc, packed = uv2, uc2, packed2
                self.repaired_count += 1

        self.decodes_total += decodes
        f1, f2 = self._objectives(packed)
        out["F"] = [f1, f2]
        out["G"] = [uv, float(uc)]
        self.eval_time_s += time.perf_counter() - t0
