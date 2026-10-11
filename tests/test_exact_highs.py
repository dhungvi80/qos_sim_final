"""
tests/test_exact_highs.py

Checks exact_solver_highs (ILP via HiGHS) against an independent exact method
(plain recursion for max-weight independent set) and basic invariants.
"""
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from qos_sim.pruning import Request, Variable
from qos_sim.qubo_builder import build_qubo
from qos_sim.baseline import (exact_solver_highs, greedy_schedule,
                              random_init_and_repair)
from qos_sim.repair import repair, verify_feasible


def _mwis(weights, conflicts):
    n = len(weights)
    adj = [0] * n
    for a, b in conflicts:
        adj[a] |= 1 << b
        adj[b] |= 1 << a
    best = 0.0

    def rec(i, mask, total):
        nonlocal best
        if i == n:
            best = max(best, total)
            return
        if total + sum(weights[i:]) <= best:
            return
        if not (adj[i] & mask):
            rec(i + 1, mask | (1 << i), total + weights[i])
        rec(i + 1, mask, total)

    rec(0, 0, 0.0)
    return best


def _random_instance(seed, n=18, p_conflict=0.2):
    rng = random.Random(seed)
    requests = [Request(id=f"r{i}", priority_weight=rng.uniform(1, 10)) for i in range(n)]
    variables = [Variable(r.id, f"p{i}", 0) for i, r in enumerate(requests)]
    conflicts = [(a, b) for a in range(n) for b in range(a + 1, n) if rng.random() < p_conflict]
    return requests, variables, conflicts


def test_matches_independent_optimum():
    for seed in range(12):
        requests, variables, conflicts = _random_instance(seed)
        qubo = build_qubo(variables, requests, conflicts)
        res = exact_solver_highs(qubo, requests, conflicts)
        w = [r.priority_weight for r in requests]
        assert res.optimal
        assert verify_feasible(res.x, conflicts)
        assert abs(res.served_utility - _mwis(w, conflicts)) < 1e-6, seed


def test_dominates_heuristics():
    for seed in range(12):
        requests, variables, conflicts = _random_instance(seed, n=24, p_conflict=0.25)
        qubo = build_qubo(variables, requests, conflicts)
        opt = exact_solver_highs(qubo, requests, conflicts).served_utility
        assert opt >= greedy_schedule(qubo, requests, conflicts).served_utility - 1e-9
        assert opt >= random_init_and_repair(qubo, requests, conflicts, seed=seed).served_utility - 1e-9
        x_raw = np.array([random.Random(seed).random() < 0.5 for _ in range(len(variables))], float)
        assert opt >= qubo_utility(qubo, repair(qubo, x_raw, requests, conflicts)) - 1e-9


def qubo_utility(qubo, x):
    from qos_sim.qubo_builder import served_utility
    return served_utility(qubo, x)


def test_no_conflicts_takes_everything():
    requests, variables, _ = _random_instance(1, n=6)
    qubo = build_qubo(variables, requests, [])
    res = exact_solver_highs(qubo, requests, [])
    assert res.optimal and res.x.sum() == 6
    assert abs(res.served_utility - sum(r.priority_weight for r in requests)) < 1e-9


def test_empty_instance():
    qubo = build_qubo([], [], [])
    res = exact_solver_highs(qubo, [], [])
    assert res.optimal and res.served_utility == 0.0 and len(res.x) == 0


def test_time_limit_reports_not_optimal_instead_of_gap_zero():
    # 1 ms limit on a larger instance: the result must either be proven optimal
    # or flagged optimal=False (never silently treated as optimal).
    requests, variables, conflicts = _random_instance(3, n=300, p_conflict=0.05)
    qubo = build_qubo(variables, requests, conflicts)
    res = exact_solver_highs(qubo, requests, conflicts, time_limit_s=0.001)
    if not res.optimal:
        assert (np.isnan(res.served_utility)) or verify_feasible(res.x, conflicts)


def test_bound_equals_optimum_when_optimal():
    for seed in range(12):
        requests, variables, conflicts = _random_instance(seed)
        qubo = build_qubo(variables, requests, conflicts)
        res = exact_solver_highs(qubo, requests, conflicts)
        w = [r.priority_weight for r in requests]
        assert res.optimal
        assert abs(res.bound - res.served_utility) < 1e-9
        assert abs(res.bound - _mwis(w, conflicts)) < 1e-6, seed


def test_bound_is_valid_upper_bound_on_timeout():
    # (n=100, p=0.1, seed 3): proven optimum is found by a long run; a very short
    # run times out. Its bound must still be >= the true optimum >= its incumbent.
    requests, variables, conflicts = _random_instance(3, n=100, p_conflict=0.1)
    qubo = build_qubo(variables, requests, conflicts)
    opt = exact_solver_highs(qubo, requests, conflicts, time_limit_s=60.0)
    assert opt.optimal
    short = exact_solver_highs(qubo, requests, conflicts, time_limit_s=0.01)
    if short.optimal:       # very fast machine: nothing to check beyond tightness
        assert abs(short.bound - opt.served_utility) < 1e-6
        return
    assert np.isfinite(short.bound)
    assert short.bound >= opt.served_utility - 1e-6
    if np.isfinite(short.served_utility):
        assert short.served_utility <= opt.served_utility + 1e-6
        assert short.bound >= short.served_utility - 1e-9


def test_empty_and_no_conflict_bound():
    assert exact_solver_highs(build_qubo([], [], []), [], []).bound == 0.0
    requests, variables, _ = _random_instance(1, n=6)
    res = exact_solver_highs(build_qubo(variables, requests, []), requests, [])
    assert abs(res.bound - sum(r.priority_weight for r in requests)) < 1e-9


if __name__ == "__main__":
    test_bound_equals_optimum_when_optimal()
    test_bound_is_valid_upper_bound_on_timeout()
    test_empty_and_no_conflict_bound()
    test_matches_independent_optimum()
    test_dominates_heuristics()
    test_no_conflicts_takes_everything()
    test_empty_instance()
    test_time_limit_reports_not_optimal_instead_of_gap_zero()
    print("All tests passed.")
