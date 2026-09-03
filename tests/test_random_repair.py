"""Unit test for random_init_and_repair ablation baseline."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from qos_sim.pruning import Request, Variable
from qos_sim.qubo_builder import QUBOInstance
from qos_sim.baseline import random_init_and_repair, greedy_schedule
from qos_sim.repair import verify_feasible


def _make_toy():
    requests = [
        Request(id="r0", priority_weight=10.0),
        Request(id="r1", priority_weight=7.0),
        Request(id="r2", priority_weight=5.0),
    ]
    variables = [
        Variable(request_id="r0", path_id="p0", time_slot=0),
        Variable(request_id="r0", path_id="p1", time_slot=0),
        Variable(request_id="r1", path_id="p0", time_slot=0),
        Variable(request_id="r1", path_id="p1", time_slot=0),
        Variable(request_id="r2", path_id="p0", time_slot=0),
        Variable(request_id="r2", path_id="p1", time_slot=0),
    ]
    conflicts = [(0, 2), (0, 4), (2, 4), (1, 3), (1, 5), (3, 5),
                 (0, 1), (2, 3), (4, 5)]
    n = len(variables)
    Q = np.zeros((n, n))
    w = {"r0": 10.0, "r1": 7.0, "r2": 5.0}
    for i, v in enumerate(variables):
        Q[i, i] = -w[v.request_id]
    for a, b in conflicts:
        Q[a, b] += 20.0
        Q[b, a] += 20.0
    index_of = {v.key(): i for i, v in enumerate(variables)}
    qubo = QUBOInstance(
        Q=Q, index_of=index_of, variables=variables,
        lambda_used=20.0, lambda_star=20.0,
    )
    return qubo, requests, conflicts


def test_random_repair_always_feasible():
    qubo, requests, conflicts = _make_toy()
    for seed in range(30):
        r = random_init_and_repair(qubo, requests, conflicts, seed=seed)
        assert r.feasible
        assert verify_feasible(r.x, conflicts)
        assert r.served_utility >= 0


def test_random_repair_not_above_greedy():
    """Completion is priority-greedy; Random+Repair should not exceed Greedy."""
    qubo, requests, conflicts = _make_toy()
    gr = greedy_schedule(qubo, requests, conflicts)
    for seed in range(30):
        r = random_init_and_repair(qubo, requests, conflicts, seed=seed)
        assert r.served_utility <= gr.served_utility + 1e-9


if __name__ == "__main__":
    test_random_repair_always_feasible()
    test_random_repair_not_above_greedy()
    print("test_random_repair: 2/2 passed")
