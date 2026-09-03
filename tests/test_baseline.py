import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from qos_sim.pruning import Request, Variable
from qos_sim.qubo_builder import build_qubo, conflict_pairs_from_paths, served_utility
from qos_sim.baseline import (greedy_schedule, simulated_annealing,
                                tabu_search, genetic_algorithm, ant_colony_optimization)


def _toy_instance():
    variables = [
        Variable("r0", "pA", 0), Variable("r1", "pB", 0),
        Variable("r2", "pC", 0), Variable("r3", "pA", 1),
    ]
    requests = [Request("r0", 9.0), Request("r1", 7.0),
                Request("r2", 5.0), Request("r3", 3.0)]
    path_resources = {"pA": ("res1",), "pB": ("res1",), "pC": ("res2",)}
    conflicts = conflict_pairs_from_paths(variables, path_resources)
    qubo = build_qubo(variables, requests, conflicts)
    return qubo, requests, conflicts


def test_greedy_always_feasible():
    qubo, requests, conflicts = _toy_instance()
    result = greedy_schedule(qubo, requests, conflicts)
    assert result.feasible
    for a, b in conflicts:
        assert not (result.x[a] > 0.5 and result.x[b] > 0.5)


def test_greedy_prefers_higher_priority():
    """r0 (w=9) và r1 (w=7) xung đột (cùng res1) -> greedy phải chọn r0."""
    qubo, requests, conflicts = _toy_instance()
    result = greedy_schedule(qubo, requests, conflicts)
    assert result.x[0] == 1.0  # r0, weight cao nhất trong nhóm xung đột
    assert result.x[1] == 0.0  # r1 bị loại vì xung đột với r0


def test_sa_runs_and_returns_valid_bitstring():
    qubo, requests, conflicts = _toy_instance()
    result = simulated_annealing(qubo, conflicts, n_iters=200, seed=1)
    assert result.x.shape == (4,)
    assert set(result.x.tolist()) <= {0.0, 1.0}
    assert result.served_utility >= 0


def test_sa_improves_over_random_on_average():
    """SA nên cho kết quả >= trung bình 20 lần random bitstring (thống kê,
    không chứng minh optimal nhưng xác nhận SA có học được gì đó)."""
    qubo, requests, conflicts = _toy_instance()
    sa_result = simulated_annealing(qubo, conflicts, n_iters=500, seed=1)

    rng = np.random.default_rng(1)
    random_utils = []
    for _ in range(20):
        x = rng.integers(0, 2, size=4).astype(float)
        random_utils.append(served_utility(qubo, x))

    assert sa_result.served_utility >= np.mean(random_utils)


def test_tabu_search_runs_and_returns_valid_bitstring():
    qubo, requests, conflicts = _toy_instance()
    result = tabu_search(qubo, conflicts, n_iters=100, seed=1)
    assert result.x.shape == (4,)
    assert set(result.x.tolist()) <= {0.0, 1.0}


def test_genetic_algorithm_runs_and_returns_valid_bitstring():
    qubo, requests, conflicts = _toy_instance()
    result = genetic_algorithm(qubo, conflicts, population_size=10,
                                n_generations=15, seed=1)
    assert result.x.shape == (4,)
    assert set(result.x.tolist()) <= {0.0, 1.0}


def test_aco_runs_and_returns_valid_bitstring():
    qubo, requests, conflicts = _toy_instance()
    result = ant_colony_optimization(qubo, conflicts, n_ants=10,
                                      n_iterations=15, seed=1)
    assert result.x.shape == (4,)
    assert set(result.x.tolist()) <= {0.0, 1.0}


def test_all_baselines_beat_or_match_random_on_toy_instance():
    qubo, requests, conflicts = _toy_instance()
    rng = np.random.default_rng(1)
    random_utils = [served_utility(qubo, rng.integers(0, 2, size=4).astype(float))
                     for _ in range(30)]
    baseline_avg = np.mean(random_utils)

    results = {
        'greedy': greedy_schedule(qubo, requests, conflicts).served_utility,
        'sa': simulated_annealing(qubo, conflicts, n_iters=300, seed=1).served_utility,
        'tabu': tabu_search(qubo, conflicts, n_iters=100, seed=1).served_utility,
        'ga': genetic_algorithm(qubo, conflicts, population_size=10,
                                 n_generations=15, seed=1).served_utility,
        'aco': ant_colony_optimization(qubo, conflicts, n_ants=10,
                                        n_iterations=15, seed=1).served_utility,
    }
    for name, u in results.items():
        assert u >= baseline_avg, f"{name}: {u} < random baseline {baseline_avg}"


if __name__ == "__main__":
    test_greedy_always_feasible()
    test_greedy_prefers_higher_priority()
    test_sa_runs_and_returns_valid_bitstring()
    test_sa_improves_over_random_on_average()
    test_tabu_search_runs_and_returns_valid_bitstring()
    test_genetic_algorithm_runs_and_returns_valid_bitstring()
    test_aco_runs_and_returns_valid_bitstring()
    test_all_baselines_beat_or_match_random_on_toy_instance()
    print("All tests passed.")
