import unittest
import numpy as np
from core.placement.rotation import RotationEngine
from core.placement.extreme_point import ExtremePointDecoder
from core.benchmarks.generator import HyperDockBenchGenerator
from core.placement.baselines import FirstFitPacker

class TestPhase1B(unittest.TestCase):

    def test_rotation_engine(self):
        dims = [10, 20, 30]
        r1 = RotationEngine.get_rotation(dims, 1)
        self.assertEqual(r1, [10, 20, 30])
        
        r2 = RotationEngine.get_rotation(dims, 2)
        self.assertEqual(r2, [10, 30, 20])
        
        r6 = RotationEngine.get_rotation(dims, 6)
        self.assertEqual(r6, [30, 20, 10])
        
        all_rots = RotationEngine.all_rotations(dims)
        self.assertEqual(len(all_rots), 6)

    def test_ep_decoder_boundary(self):
        decoder = ExtremePointDecoder([10.0, 10.0, 10.0])
        # Fit perfectly
        self.assertTrue(decoder.can_pack((0,0,0), [10, 10, 10]))
        # Exceed boundaries
        self.assertFalse(decoder.can_pack((0,0,0), [11, 10, 10]))
        self.assertFalse(decoder.can_pack((1,0,0), [10, 10, 10]))

    def test_ep_decoder_collision(self):
        decoder = ExtremePointDecoder([20.0, 20.0, 20.0])
        decoder.add_item((0,0,0), [10, 10, 10], {"id": 1})
        
        # Exact overlap
        self.assertFalse(decoder.can_pack((0,0,0), [5, 5, 5]))
        # Partial overlap
        self.assertFalse(decoder.can_pack((5,5,5), [10, 10, 10]))
        # Next to it (x axis)
        self.assertTrue(decoder.can_pack((10,0,0), [10, 10, 10]))

    def test_dataset_determinism(self):
        g1 = HyperDockBenchGenerator("HB-TEST-50", "SC1", 50)
        d1 = g1.generate()
        
        g2 = HyperDockBenchGenerator("HB-TEST-50", "SC1", 50)
        d2 = g2.generate()
        
        self.assertEqual(d1["seed"], d2["seed"])
        self.assertEqual(d1["items"][0]["dimensions_wdh_meters"], d2["items"][0]["dimensions_wdh_meters"])

    def test_baseline_integration(self):
        g = HyperDockBenchGenerator("HB-INT-10", "SC1", 10)
        data = g.generate()
        
        c = data["containers"][0]["dimensions_wdh_meters"]
        items = data["items"]
        
        packer = FirstFitPacker(c)
        packed_count, packed_vol = packer.pack(items)
        
        self.assertTrue(packed_count >= 0)
        self.assertTrue(packed_vol >= 0.0)

if __name__ == '__main__':
    unittest.main()
