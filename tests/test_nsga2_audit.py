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
        
    sampler = CargoSampling(n_items=DummyProblem.n_items)
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
        
    sampler = CargoSampling(n_items=DummyProblem.n_items)
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
    """Unplaced items must register as constraint violations (G > 0), not as objective values."""
    items = [
        {"item_id": "1", "dimensions_wdh_meters": [1.0, 1.0, 1.0], "mass_kg": 5000.0},
        {"item_id": "2", "dimensions_wdh_meters": [1.0, 1.0, 1.0], "mass_kg": 5000.0} # Too big for container
    ]
    # Container is exactly 1x1x1, so only 1 item can fit
    problem = CargoProblem(items=items, container_dims=[1.0, 1.0, 1.0])
    
    x = np.array([0, 1, 1, 1])
    out = {}
    problem._evaluate(x, out)
    
    # Current design: two objectives and two inequality constraints (g1 = unplaced volume, g2 = unplaced count)
    assert len(out["F"]) == 2
    assert len(out["G"]) == 2
    g1, g2 = out["G"]
    
    # One item (volume 1.0) cannot be placed
    assert g1 == 1.0
    assert g2 == 1
    assert np.all(np.array(out["G"]) > 0)  # infeasible
    assert np.all(np.isfinite(np.array(out["F"])))

def test_feasible_solution_scoring():
    """A fully packed layout has no constraint violation and finite objectives."""
    items = [
        {"item_id": "1", "dimensions_wdh_meters": [0.5, 0.5, 0.5], "mass_kg": 10.0}
    ]
    problem = CargoProblem(items=items, container_dims=[1.0, 1.0, 1.0])
    
    x = np.array([0, 1])
    out = {}
    problem._evaluate(x, out)
    
    assert len(out["F"]) == 2
    
    # No unplaced volume or count: feasible
    assert np.all(np.array(out["G"]) <= 0)
    
    f1, f2 = out["F"]
    # f1 = CoM deviation, should be well under 100 m for a single small item
    assert 0.0 <= f1 < 100.0
    # f2 = -mean CASC, so it lies in [-1, 0]
    assert -1.0 <= f2 <= 0.0
