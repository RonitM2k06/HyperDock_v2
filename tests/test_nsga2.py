import pytest
import numpy as np
from pymoo.core.problem import ElementwiseProblem
from optimization.nsga2.operators import CargoCrossover, CargoMutation

# Mock problem for testing operators
class MockProblem(ElementwiseProblem):
    def __init__(self, n_var=10):
        super().__init__(n_var=n_var, n_obj=3)

def test_crossover_maintains_permutation():
    n_items = 5
    prob = MockProblem(n_var=2 * n_items)
    
    # Create two parents
    # p1 sequence: [0, 1, 2, 3, 4], orientations: [1, 1, 1, 1, 1]
    p1 = np.array([0, 1, 2, 3, 4, 1, 1, 1, 1, 1])
    # p2 sequence: [4, 3, 2, 1, 0], orientations: [6, 6, 6, 6, 6]
    p2 = np.array([4, 3, 2, 1, 0, 6, 6, 6, 6, 6])
    
    X = np.stack([p1, p2]).reshape(2, 1, 2 * n_items)
    
    crossover = CargoCrossover(n_items=n_items, prob=1.0)
    
    # We set random seed to ensure deterministic behavior for this test
    np.random.seed(42)
    Y = crossover._do(prob, X)
    
    o1, o2 = Y[0, 0], Y[1, 0]
    
    # Verify sequence part (first 5 elements) is a valid permutation of [0, 1, 2, 3, 4]
    assert np.all(np.sort(o1[:n_items]) == np.arange(n_items))
    assert np.all(np.sort(o2[:n_items]) == np.arange(n_items))
    
    # Verify orientations (last 5 elements) are valid values between 1 and 6
    assert np.all((o1[n_items:] >= 1) & (o1[n_items:] <= 6))
    assert np.all((o2[n_items:] >= 1) & (o2[n_items:] <= 6))

def test_mutation_maintains_validity():
    n_items = 5
    prob = MockProblem(n_var=2 * n_items)
    
    X = np.array([[0, 1, 2, 3, 4, 1, 1, 1, 1, 1]])
    
    mutation = CargoMutation(n_items=n_items, prob=1.0) # 100% mutation chance
    
    np.random.seed(42)
    Y = mutation._do(prob, X)
    
    o = Y[0]
    
    # Verify sequence is a valid permutation
    assert np.all(np.sort(o[:n_items]) == np.arange(n_items))
    
    # Verify orientations are mutated but valid (1 to 6)
    assert np.all((o[n_items:] >= 1) & (o[n_items:] <= 6))
    
    # Ensure it actually mutated something
    assert not np.array_equal(X[0], o)

def test_deterministic_seed_integration():
    """Test that applying crossover and mutation with a fixed seed produces repeatable results."""
    n_items = 10
    prob = MockProblem(n_var=2 * n_items)
    
    X = np.zeros((2, 1, 2 * n_items), dtype=int)
    X[0, 0] = np.concatenate([np.arange(n_items), np.ones(n_items, dtype=int)])
    X[1, 0] = np.concatenate([np.arange(n_items)[::-1], np.full(n_items, 6, dtype=int)])
    
    crossover = CargoCrossover(n_items=n_items, prob=1.0)
    mutation = CargoMutation(n_items=n_items, prob=1.0)
    
    np.random.seed(123)
    Y1 = crossover._do(prob, X)
    Y1_mut = mutation._do(prob, Y1.reshape(2, 2 * n_items))
    
    np.random.seed(123)
    Y2 = crossover._do(prob, X)
    Y2_mut = mutation._do(prob, Y2.reshape(2, 2 * n_items))
    
    assert np.array_equal(Y1_mut, Y2_mut)
