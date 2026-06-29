import pytest
import numpy as np
from core.metrics.casc import CASCv2, CASCv2Config

@pytest.fixture
def casc():
    config = CASCv2Config(L_arm=0.75, gamma=2.0, lambda_occ=1.0, C_grasp=2.0, kappa=0.05, grid_resolution=10)
    return CASCv2(config)

def test_reach_penalty_monotonicity(casc):
    """Items further back should have a lower CASC score (higher reach penalty)."""
    item1 = {
        'point': (0, 0, 0.1), # z=0.1
        'dimensions': (0.2, 0.2, 0.2),
        'data': {'item_id': 'near', 'category': 'hardware'}
    }
    item2 = {
        'point': (0.5, 0, 0.8), # z=0.8, past L_arm
        'dimensions': (0.2, 0.2, 0.2),
        'data': {'item_id': 'far', 'category': 'hardware'}
    }
    
    res1 = casc.evaluate_layout([item1])['item_details'][0]
    res2 = casc.evaluate_layout([item2])['item_details'][0]
    
    assert res1['c_reach'] < res2['c_reach']
    assert res1['casc_score'] > res2['casc_score']

def test_occlusion_penalty(casc):
    """An item that is blocked should have a higher occlusion penalty."""
    target = {
        'point': (0, 0, 0.5),
        'dimensions': (0.5, 0.5, 0.5),
        'data': {'item_id': 'target', 'category': 'hardware'}
    }
    blocker = {
        'point': (0.1, 0.1, 0.2), # In front of target, overlapping XY
        'dimensions': (0.2, 0.2, 0.2),
        'data': {'item_id': 'blocker', 'category': 'hardware'}
    }
    non_blocker = {
        'point': (0.6, 0.6, 0.2), # In front, but NO XY overlap
        'dimensions': (0.2, 0.2, 0.2),
        'data': {'item_id': 'non_blocker', 'category': 'hardware'}
    }
    
    res_occluded = casc.evaluate_layout([target, blocker])
    target_occ_res = next(x for x in res_occluded['item_details'] if x['item_id'] == 'target')
    
    res_clear = casc.evaluate_layout([target, non_blocker])
    target_clear_res = next(x for x in res_clear['item_details'] if x['item_id'] == 'target')
    
    assert target_occ_res['c_occ'] > target_clear_res['c_occ']
    assert target_clear_res['c_occ'] == 0.0

def test_extraction_mass_and_rigidity(casc):
    """Tests that moving many light items is worse than one heavy item, and rigidity reduces penalty."""
    target = {
        'point': (0, 0, 0.5),
        'dimensions': (0.2, 0.2, 0.2),
        'data': {'item_id': 'target', 'category': 'hardware'}
    }
    
    # 1 blocker of 10kg (Hard)
    blocker_heavy = {
        'point': (0, 0, 0.2),
        'dimensions': (0.2, 0.2, 0.2),
        'data': {'item_id': 'b1', 'category': 'hardware', 'mass_kg': 10.0}
    }
    
    # 5 blockers of 2kg (Hard)
    blockers_light = [
        {
            'point': (0, 0, 0.1 + 0.05 * i),
            'dimensions': (0.2, 0.2, 0.05),
            'data': {'item_id': f'l{i}', 'category': 'hardware', 'mass_kg': 2.0}
        } for i in range(5)
    ]
    
    # 1 blocker of 10kg (Soft)
    blocker_soft = {
        'point': (0, 0, 0.2),
        'dimensions': (0.2, 0.2, 0.2),
        'data': {'item_id': 'b_soft', 'category': 'consumable', 'mass_kg': 10.0}
    }
    
    res_heavy = casc.evaluate_layout([target, blocker_heavy])
    c_ext_heavy = next(x for x in res_heavy['item_details'] if x['item_id'] == 'target')['c_ext']
    
    res_light = casc.evaluate_layout([target] + blockers_light)
    c_ext_light = next(x for x in res_light['item_details'] if x['item_id'] == 'target')['c_ext']
    
    res_soft = casc.evaluate_layout([target, blocker_soft])
    c_ext_soft = next(x for x in res_soft['item_details'] if x['item_id'] == 'target')['c_ext']
    
    assert c_ext_light > c_ext_heavy
    assert c_ext_soft < c_ext_heavy

def test_blocker_count_monotonicity(casc):
    """More blockers must strictly increase the extraction penalty."""
    target = {
        'point': (0, 0, 0.5),
        'dimensions': (0.2, 0.2, 0.2),
        'data': {'item_id': 'target', 'category': 'hardware'}
    }
    b1 = {'point': (0, 0, 0.1), 'dimensions': (0.2, 0.2, 0.1), 'data': {'item_id': 'b1', 'category': 'hardware', 'mass_kg': 2.0}}
    b2 = {'point': (0, 0, 0.2), 'dimensions': (0.2, 0.2, 0.1), 'data': {'item_id': 'b2', 'category': 'hardware', 'mass_kg': 2.0}}
    
    res1 = casc.evaluate_layout([target, b1])
    target_1_b = next(x for x in res1['item_details'] if x['item_id'] == 'target')
    
    res2 = casc.evaluate_layout([target, b1, b2])
    target_2_b = next(x for x in res2['item_details'] if x['item_id'] == 'target')
    
    assert target_2_b['c_ext'] > target_1_b['c_ext']
    assert target_2_b['casc_score'] < target_1_b['casc_score']

def test_visibility_monotonicity(casc):
    """An item that is half-blocked should have a higher score than a fully blocked item."""
    target = {
        'point': (0, 0, 0.5),
        'dimensions': (0.4, 0.4, 0.4),
        'data': {'item_id': 'target', 'category': 'hardware'}
    }
    half_blocker = {
        'point': (0, 0, 0.2),
        'dimensions': (0.2, 0.4, 0.2), # Blocks half of the 0.4x0.4 face
        'data': {'item_id': 'hb', 'category': 'hardware'}
    }
    full_blocker = {
        'point': (0, 0, 0.2),
        'dimensions': (0.4, 0.4, 0.2), # Blocks full face
        'data': {'item_id': 'fb', 'category': 'hardware'}
    }
    
    res_half = casc.evaluate_layout([target, half_blocker])
    target_half = next(x for x in res_half['item_details'] if x['item_id'] == 'target')
    
    res_full = casc.evaluate_layout([target, full_blocker])
    target_full = next(x for x in res_full['item_details'] if x['item_id'] == 'target')
    
    assert target_half['c_occ'] < target_full['c_occ']
    assert target_half['casc_score'] > target_full['casc_score']

def test_fully_occluded_item(casc):
    """A fully occluded item should NOT drop its total CASC to 0 (tests removal of fatal product formulation)."""
    target = {
        'point': (0, 0, 0.1), # Very close
        'dimensions': (0.2, 0.2, 0.2),
        'data': {'item_id': 'target', 'category': 'hardware'}
    }
    full_blocker = {
        'point': (0, 0, 0.0),
        'dimensions': (0.2, 0.2, 0.05),
        'data': {'item_id': 'fb', 'category': 'hardware'}
    }
    
    res = casc.evaluate_layout([target, full_blocker])
    target_res = next(x for x in res['item_details'] if x['item_id'] == 'target')
    
    # Visually fully occluded
    assert target_res['c_occ'] == casc.config.lambda_occ
    # BUT CASC must not be 0
    assert target_res['casc_score'] > 0.0

def test_no_blocker_case(casc):
    """An item with zero blockers should have 0 extraction and 0 occlusion penalties."""
    target = {
        'point': (0, 0, 0.5),
        'dimensions': (0.2, 0.2, 0.2),
        'data': {'item_id': 'target', 'category': 'hardware'}
    }
    res = casc.evaluate_layout([target])
    target_res = res['item_details'][0]
    
    assert target_res['c_occ'] == 0.0
    assert target_res['c_ext'] == 0.0
    assert target_res['transitive_blockers_count'] == 0

def test_parameter_sensitivity():
    """Testing that changes to parameters actually perturb the score."""
    item = {
        'point': (0, 0, 0.5),
        'dimensions': (0.2, 0.2, 0.2),
        'data': {'item_id': 'target', 'category': 'hardware'}
    }
    
    casc1 = CASCv2(CASCv2Config(L_arm=0.5))  # High reach penalty
    casc2 = CASCv2(CASCv2Config(L_arm=1.0))  # Low reach penalty
    
    res1 = casc1.evaluate_layout([item])['item_details'][0]
    res2 = casc2.evaluate_layout([item])['item_details'][0]
    
    assert res1['c_reach'] > res2['c_reach']
    assert res1['casc_score'] < res2['casc_score']
