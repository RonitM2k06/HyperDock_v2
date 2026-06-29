import numpy as np
from pymoo.core.crossover import Crossover
from pymoo.core.mutation import Mutation
from pymoo.core.sampling import Sampling

class CargoSampling(Sampling):
    """
    Custom Sampling for Cargo Optimization.
    Initializes valid permutations for the sequence part, 
    and random integers (1-6) for the orientation part.
    """
    def __init__(self, n_items: int):
        super().__init__()
        self.n = n_items

    def _do(self, problem, n_samples, **kwargs):
        X = np.empty((n_samples, 2 * self.n), dtype=int)
        for i in range(n_samples):
            # Valid permutation for items
            X[i, :self.n] = np.random.permutation(self.n)
            # Random orientations 1 to 6
            X[i, self.n:] = np.random.randint(1, 7, size=self.n)
        return X

class CargoCrossover(Crossover):
    """
    Hybrid Crossover for Cargo Optimization.
    Applies Order Crossover (OX1) to the sequence (indices 0 to N-1).
    Applies Uniform Crossover to the orientation (indices N to 2N-1).
    """
    def __init__(self, n_items: int, prob=0.9):
        # 2 parents, 2 offspring
        super().__init__(2, 2, prob=prob)
        self.n = n_items
        self.crossover_prob = prob

    def _do(self, problem, X, **kwargs):
        _, n_matings, n_var = X.shape
        Y = np.empty_like(X)

        for i in range(n_matings):
            p1, p2 = X[0, i], X[1, i]
            o1, o2 = np.copy(p1), np.copy(p2)
            
            # 1. Order Crossover (OX1) for Sequence (0 to n-1)
            seq1, seq2 = p1[:self.n], p2[:self.n]
            
            if np.random.rand() < self.crossover_prob:
                a, b = np.sort(np.random.choice(self.n, 2, replace=False))
                
                # Offspring 1
                o1_seq = np.full(self.n, -1)
                o1_seq[a:b+1] = seq1[a:b+1]
                p2_filtered = [x for x in seq2 if x not in o1_seq[a:b+1]]
                idx = 0
                for j in range(self.n):
                    if o1_seq[j] == -1:
                        o1_seq[j] = p2_filtered[idx]
                        idx += 1
                o1[:self.n] = o1_seq

                # Offspring 2
                o2_seq = np.full(self.n, -1)
                o2_seq[a:b+1] = seq2[a:b+1]
                p1_filtered = [x for x in seq1 if x not in o2_seq[a:b+1]]
                idx = 0
                for j in range(self.n):
                    if o2_seq[j] == -1:
                        o2_seq[j] = p1_filtered[idx]
                        idx += 1
                o2[:self.n] = o2_seq

            # 2. Uniform Crossover for Orientation (n to 2n-1)
            if np.random.rand() < self.crossover_prob:
                mask = np.random.rand(self.n) < 0.5
                
                # Swap bits where mask is True
                o1[self.n:][mask] = p2[self.n:][mask]
                o2[self.n:][mask] = p1[self.n:][mask]
                
            Y[0, i], Y[1, i] = o1, o2

        return Y

class CargoMutation(Mutation):
    """
    Hybrid Mutation for Cargo Optimization.
    Sequence: Swap & Inversion mutation.
    Orientation: Random reset mutation.
    """
    def __init__(self, n_items: int, prob=None):
        super().__init__()
        self.n = n_items
        self.prob = prob if prob is not None else 1.0 / n_items

    def _do(self, problem, X, **kwargs):
        Y = np.copy(X)
        for i in range(len(Y)):
            ind = Y[i]
            
            # 1. Sequence Mutation (Swap or Inversion)
            if np.random.rand() < self.prob:
                a, b = np.sort(np.random.choice(self.n, 2, replace=False))
                if np.random.rand() < 0.5:
                    # Swap
                    ind[a], ind[b] = ind[b], ind[a]
                else:
                    # Inversion
                    ind[a:b+1] = ind[a:b+1][::-1]
                    
            # 2. Orientation Mutation (Random Reset)
            for j in range(self.n, 2*self.n):
                if np.random.rand() < self.prob:
                    ind[j] = np.random.randint(1, 7)
                    
            Y[i] = ind
        return Y
