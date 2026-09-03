import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from qos_sim.dvqe_partition import (partition_by_weakest_edge, reconstruct_expectation,
                                      marginals_from_probabilities)


def test_cuts_weak_edges_not_strong():
    Q = np.zeros((4, 4))
    Q[0, 1] = Q[1, 0] = 10.0   # strong
    Q[2, 3] = Q[3, 2] = 10.0   # strong
    Q[1, 2] = Q[2, 1] = 0.01   # weak bridge
    parts = partition_by_weakest_edge(Q, max_size=2)
    parts_sets = [set(p) for p in parts]
    assert {0, 1} in parts_sets
    assert {2, 3} in parts_sets


def test_respects_max_size():
    n = 12
    rng = np.random.default_rng(0)
    Q = rng.random((n, n))
    Q = (Q + Q.T) / 2
    parts = partition_by_weakest_edge(Q, max_size=5)
    assert all(len(p) <= 5 for p in parts)
    assert sum(len(p) for p in parts) == n


def test_reconstruct_matches_full_expectation_when_independent():
    """If marginals are exact and Q has no correlation structure beyond
    products, reconstruct_expectation should match brute-force <H_C>
    computed from the product-form full distribution."""
    Q = np.array([[-2, 1, 0], [1, -3, 0.5], [0, 0.5, -1]])
    parts = [[0, 1], [2]]
    marg = [np.array([0.8, 0.6]), np.array([0.3])]
    val = reconstruct_expectation(Q, parts, marg)

    m = np.array([0.8, 0.6, 0.3])
    expected = sum(Q[i, i] * m[i] for i in range(3))
    expected += 2 * Q[0, 2] * m[0] * m[2] + 2 * Q[1, 2] * m[1] * m[2]
    assert abs(val - expected) < 1e-9


def test_marginals_from_probabilities_uniform():
    # uniform distribution over 2 qubits -> each marginal = 0.5
    probs = np.ones(4) / 4
    marg = marginals_from_probabilities(probs, 2)
    assert np.allclose(marg, [0.5, 0.5])


def test_marginals_from_probabilities_deterministic():
    # state |11> with probability 1 -> both qubits marginal = 1.0
    probs = np.zeros(4)
    probs[0b11] = 1.0
    marg = marginals_from_probabilities(probs, 2)
    assert np.allclose(marg, [1.0, 1.0])


if __name__ == "__main__":
    test_cuts_weak_edges_not_strong()
    test_respects_max_size()
    test_reconstruct_matches_full_expectation_when_independent()
    test_marginals_from_probabilities_uniform()
    test_marginals_from_probabilities_deterministic()
    print("All tests passed.")
