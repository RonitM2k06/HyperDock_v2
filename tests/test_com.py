import pytest
import numpy as np
from core.metrics.com import CoMModel, CoMConfig, ContainerTransform

@pytest.fixture
def com_model():
    # Zero dry mass to isolate cargo CoM effects
    config = CoMConfig(dry_mass=0.0, target_com=np.array([0.0, 0.0, 0.0]))
    return CoMModel(config)

def test_single_mass_center(com_model):
    """Test absolute CoM for a single item matches its geometric center offset."""
    item = {
        'point': (0.0, 0.0, 0.0),
        'dimensions': (2.0, 2.0, 2.0),
        'data': {'mass_kg': 100.0}
    }
    
    res = com_model.evaluate_layout([item])
    
    # Expected: local CoM is w/2, d/2, h/2 = 1, 1, 1
    assert np.allclose(res['system_com'], [1.0, 1.0, 1.0])
    assert res['total_mass'] == 100.0

def test_symmetric_balance(com_model):
    """Test two identical items symmetrically placed cancel out their offsets."""
    # Place items to be symmetric around x=1, y=1, z=0.5
    item1 = {
        'point': (0.0, 0.0, 0.0),
        'dimensions': (1.0, 2.0, 1.0),
        'data': {'mass_kg': 50.0}
    }
    item2 = {
        'point': (1.0, 0.0, 0.0),
        'dimensions': (1.0, 2.0, 1.0),
        'data': {'mass_kg': 50.0}
    }
    
    res = com_model.evaluate_layout([item1, item2])
    
    # CoM of item1 = [0.5, 1.0, 0.5]
    # CoM of item2 = [1.5, 1.0, 0.5]
    # Average = [1.0, 1.0, 0.5]
    assert np.allclose(res['system_com'], [1.0, 1.0, 0.5])

def test_mass_weighting(com_model):
    """Test CoM is pulled toward the heavier item."""
    item1 = {
        'point': (0.0, 0.0, 0.0),
        'dimensions': (1.0, 1.0, 1.0),
        'data': {'mass_kg': 10.0}
    } # CoM: 0.5, 0.5, 0.5
    item2 = {
        'point': (1.0, 0.0, 0.0),
        'dimensions': (1.0, 1.0, 1.0),
        'data': {'mass_kg': 90.0}
    } # CoM: 1.5, 0.5, 0.5
    
    res = com_model.evaluate_layout([item1, item2])
    
    # Expected X: (10 * 0.5 + 90 * 1.5) / 100 = 1.4
    assert np.allclose(res['system_com'], [1.4, 0.5, 0.5])

def test_multi_container_computation():
    """Test transformations mapping local container coordinates to absolute SCS."""
    config = CoMConfig(dry_mass=0.0, target_com=np.array([0.0, 0.0, 0.0]))
    
    transforms = {
        "C1": ContainerTransform(container_id="C1", R_M=np.array([10.0, 0.0, 0.0])),
        "C2": ContainerTransform(container_id="C2", R_M=np.array([-10.0, 0.0, 0.0]))
    }
    model = CoMModel(config, transforms)
    
    item1 = {
        'point': (0.0, 0.0, 0.0),
        'dimensions': (2.0, 2.0, 2.0),
        'data': {'mass_kg': 100.0, 'container_id': 'C1'}
    }
    item2 = {
        'point': (0.0, 0.0, 0.0),
        'dimensions': (2.0, 2.0, 2.0),
        'data': {'mass_kg': 100.0, 'container_id': 'C2'}
    }
    
    res = model.evaluate_layout([item1, item2])
    
    # C1 abs: 10 + 1 = 11
    # C2 abs: -10 + 1 = -9
    # Average X: (11 + (-9)) / 2 = 1.0
    assert np.allclose(res['system_com'], [1.0, 1.0, 1.0])

def test_target_com_deviation():
    """Test Euclidean deviation from a specific target CoM."""
    config = CoMConfig(dry_mass=0.0, target_com=np.array([5.0, 5.0, 5.0]))
    model = CoMModel(config)
    
    item = {
        'point': (0.0, 0.0, 0.0),
        'dimensions': (2.0, 2.0, 2.0),
        'data': {'mass_kg': 100.0}
    }
    
    res = model.evaluate_layout([item])
    # System CoM = [1, 1, 1]
    # Target = [5, 5, 5]
    # Diff = [-4, -4, -4] -> length = sqrt(16+16+16) = sqrt(48) ~ 6.928
    expected_dev = np.sqrt(48)
    assert np.isclose(res['delta_com'], expected_dev)

def test_ellipsoidal_deviation():
    """Test ellipsoidal deviation where axes have different tolerances."""
    config = CoMConfig(
        dry_mass=0.0, 
        target_com=np.array([1.0, 1.0, 1.0]),
        epsilon_x=0.05, 
        epsilon_y=0.10, 
        epsilon_z=0.20
    )
    model = CoMModel(config)
    
    # Place item such that system CoM is [1.05, 1.10, 1.20]
    # Which corresponds exactly to 1 epsilon unit off in each axis
    item = {
        'point': (1.05 - 0.5, 1.10 - 0.5, 1.20 - 0.5), # shift back by local center
        'dimensions': (1.0, 1.0, 1.0),
        'data': {'mass_kg': 100.0}
    }
    
    res = model.evaluate_layout([item])
    
    # Diff = [0.05, 0.10, 0.20]
    # Ellipsoid dev = sqrt((0.05/0.05)^2 + (0.10/0.10)^2 + (0.20/0.20)^2) = sqrt(1+1+1) = sqrt(3) ~ 1.732
    assert np.isclose(res['ellipsoidal_dev'], np.sqrt(3))

def test_translation_invariance(com_model):
    """Test that shifting the entire coordinate frame by D shifts the CoM by exactly D."""
    item_base = {
        'point': (0.0, 0.0, 0.0),
        'dimensions': (1.0, 1.0, 1.0),
        'data': {'mass_kg': 10.0}
    }
    
    item_shifted = {
        'point': (10.0, -5.0, 2.5),
        'dimensions': (1.0, 1.0, 1.0),
        'data': {'mass_kg': 10.0}
    }
    
    res_base = com_model.evaluate_layout([item_base])
    res_shifted = com_model.evaluate_layout([item_shifted])
    
    diff = np.array(res_shifted['system_com']) - np.array(res_base['system_com'])
    assert np.allclose(diff, [10.0, -5.0, 2.5])

def test_zero_mass_edge_case():
    """Test behavior when total mass is 0 (should safely default)."""
    config = CoMConfig(dry_mass=0.0, target_com=np.array([1.0, 2.0, 3.0]))
    model = CoMModel(config)
    
    item = {
        'point': (0.0, 0.0, 0.0),
        'dimensions': (1.0, 1.0, 1.0),
        'data': {'mass_kg': 0.0} # Zero mass item
    }
    
    res = model.evaluate_layout([item])
    
    assert res['total_mass'] == 0.0
    # Should safely return zeros instead of divide-by-zero NaN
    assert np.allclose(res['system_com'], [0.0, 0.0, 0.0])
