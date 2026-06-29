import pytest
import numpy as np
from optimization.nsga2.operators import CargoCrossover, CargoMutation
from optimization.nsga2.problem import CargoProblem
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

def test_chromosome_validity():
    """Test that generated chromosomes from sampling have valid permutation structures."""
    from experiments.nsga2_runner import CargoSampling
    class DummyProblem:
        n_items = 5
        n_var = 10
        
    sampler = CargoSampling()
    X = sampler._do(DummyProblem(), 10)
    
    for x in X:
        sequence = x[:5]
        # Must be a perfect permutation of 0..4
        assert set(sequence) == set(range(5))
        assert len(set(sequence)) == 5

def test_orientation_bounds():
    """Test that all orientations are strictly within [1, 6]."""
    from experiments.nsga2_runner import CargoSampling
    class DummyProblem:
        n_items = 5
        n_var = 10
        
    sampler = CargoSampling()
    X = sampler._do(DummyProblem(), 100)
    
    orientations = X[:, 5:]
    assert np.all(orientations >= 1)
    assert np.all(orientations <= 6)

def test_pareto_sorting():
    """Test that standard Fast Non-Dominated Sorting correctly identifies Pareto fronts."""
    # F contains 3 objectives for 4 solutions
    # Sol 1 dominates Sol 2
    # Sol 3 and Sol 4 are mutually non-dominating
    F = np.array([
        [1.0, 1.0, 1.0], # Sol 0 (Dominates Sol 1)
        [2.0, 2.0, 2.0], # Sol 1 (Dominated by everyone)
        [0.5, 2.0, 0.5], # Sol 2 (Non-dominated)
        [2.0, 0.5, 0.5], # Sol 3 (Non-dominated)
    ])
    
    nds = NonDominatedSorting()
    fronts = nds.do(F)
    
    # Front 0 should contain Sol 0, Sol 2, Sol 3
    assert set(fronts[0]) == {0, 2, 3}
    # Front 1 should contain Sol 1
    assert set(fronts[1]) == {1}

def test_penalty_activation():
    """Test that penalties correctly dominate the objective when items are unpacked."""
    items = [
        {"item_id": "1", "dimensions_wdh_meters": [1.0, 1.0, 1.0], "mass_kg": 5000.0},
        {"item_id": "2", "dimensions_wdh_meters": [1.0, 1.0, 1.0], "mass_kg": 5000.0} # Too big for container
    ]
    # Container is exactly 1x1x1, so only 1 item can fit
    problem = CargoProblem(items=items, container_dims=[1.0, 1.0, 1.0])
    
    x = np.array([0, 1, 1, 1])
    out = {}
    problem._evaluate(x, out)
    
    f1, f2, f3 = out["F"]
    
    # f1 = unplaced volume = 1.0 (item 2)
    assert f1 == 1.0
    
    # f2 = delta_com + lambda * unplaced_mass
    # lambda is 100.0, unplaced mass is 5000.0
    # Expected f2 > 500,000
    assert f2 >= 500000.0
    
    # f3 = -casc + eta * unplaced_count
    # eta is 100.0, unpacked = 1
    # Expected f3 > 90.0
    assert f3 >= 90.0

def test_feasible_solution_scoring():
    """Test that a perfectly feasible solution has ZERO penalties."""
    items = [
        {"item_id": "1", "dimensions_wdh_meters": [0.5, 0.5, 0.5], "mass_kg": 10.0}
    ]
    problem = CargoProblem(items=items, container_dims=[1.0, 1.0, 1.0])
    
    x = np.array([0, 1])
    out = {}
    problem._evaluate(x, out)
    
    f1, f2, f3 = out["F"]
    
    # f1 = 0.0 unplaced volume
    assert f1 == 0.0
    
    # f2 = raw com deviation, should be very small, well under 100
    assert f2 < 100.0
    
    # f3 = raw CASC (negative), should be between 0 and -1.0
    assert f3 <= 0.0 and f3 >= -1.0
