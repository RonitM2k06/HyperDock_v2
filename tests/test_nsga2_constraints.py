import numpy as np
import pytest
from pymoo.core.population import Population
from pymoo.core.individual import Individual
from pymoo.util.dominator import Dominator

class MockInd:
    def __init__(self, f, cv):
        self.F = f
        self.CV = np.array([cv])
        
def create_pop(F_vals, G_vals):
    pop = []
    for i in range(len(F_vals)):
        cv = np.sum(np.maximum(0, G_vals[i]))
        pop.append(MockInd(F_vals[i], cv))
    return pop

def test_feasible_dominates_infeasible():
    """Test that a feasible solution always dominates an infeasible one."""
    F = np.array([[10.0, 10.0], [1.0, 1.0]])
    G = np.array([[-1.0, -1.0], [5.0, 2.0]])
    pop = create_pop(F, G)
    
    # 1 means pop[0] dominates pop[1]
    rel = Dominator.get_relation(pop[0].F, pop[1].F, cva=pop[0].CV[0], cvb=pop[1].CV[0])
    assert rel == 1

def test_lower_violation_dominates_higher_violation():
    """Test that among infeasible solutions, lower CV dominates higher CV."""
    F = np.array([[1.0, 1.0], [10.0, 10.0]])
    G = np.array([[5.0, 5.0], [1.0, 1.0]])
    pop = create_pop(F, G)
    
    # CV of 0 is 10.0, CV of 1 is 2.0. Pop[1] should dominate Pop[0] -> rel = -1
    rel = Dominator.get_relation(pop[0].F, pop[1].F, cva=pop[0].CV[0], cvb=pop[1].CV[0])
    assert rel == -1

def test_objective_ranking_among_feasible():
    """Test that standard Pareto sorting applies when both are feasible."""
    F = np.array([[1.0, 5.0], [5.0, 1.0], [10.0, 10.0]])
    G = np.array([[0.0, 0.0], [-1.0, -1.0], [-2.0, -2.0]])
    pop = create_pop(F, G)
    
    # Pop 0 and Pop 1 are non-dominated -> 0
    assert Dominator.get_relation(pop[0].F, pop[1].F, cva=pop[0].CV[0], cvb=pop[1].CV[0]) == 0
    # Pop 0 dominates Pop 2 -> 1
    assert Dominator.get_relation(pop[0].F, pop[2].F, cva=pop[0].CV[0], cvb=pop[2].CV[0]) == 1
