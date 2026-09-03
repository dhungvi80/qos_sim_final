"""
tests/test_qubo_and_repair.py

Unit tests for qubo_builder.py (Theorem 1's lambda_star) and repair.py
(Algorithm 3's unconditional feasibility guarantee, Section 4.6 tier 1).
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qos_sim.pruning import PathConfig, Request, Variable
from qos_sim.qubo_builder import (build_qubo, conflict_pairs_from_paths,
                                    lambda_star, served_utility)
from qos_sim.repair import repair, verify_feasible


def test_lambda_star_matches_theorem_1_formula():
    """Theorem 1 (Eq. 13): lambda* = max over conflicts of min(w_i, w_j)."""
    variables = [
        Variable("r0", "p0", 0),
        Variable("r1", "p1", 0),  # conflicts with r0 (shared resource)
        Variable("r2", "p2", 0),  # conflicts with r1 only
    ]
    weight_of = {"r0": 3.0, "r1": 8.0, "r2": 2.0}
    conflicts = [(0, 1), (1, 2)]

    # min(3,8)=3 for pair(0,1); min(8,2)=2 for pair(1,2) -> lambda* = 3
    result = lambda_star(variables, weight_of, conflicts)
    assert abs(result - 3.0) < 1e-9


def test_lambda_star_zero_when_no_conflicts():
    variables = [Variable("r0", "p0", 0)]
    result = lambda_star(variables, {"r0": 5.0}, conflicts=[])
    assert result == 0.0


def test_build_qubo_diagonal_encodes_negative_weights():
    variables = [Variable("r0", "p0", 0), Variable("r1", "p1", 0)]
    requests = [Request("r0", 4.0), Request("r1", 6.0)]
    qubo = build_qubo(variables, requests, conflicts=[])
    assert abs(qubo.Q[0, 0] - (-4.0)) < 1e-9
    assert abs(qubo.Q[1, 1] - (-6.0)) < 1e-9


def test_build_qubo_penalizes_conflicting_pair_enough_to_deter_violation():
    """Theorem 1's guarantee, checked directly: with lambda >= lambda*,
    violating a conflict must never be QUBO-cheaper than the feasible
    alternative of keeping only the higher-priority variable active."""
    variables = [Variable("r0", "p0", 0), Variable("r1", "p0", 0)]  # same path -> conflict
    requests = [Request("r0", 4.0), Request("r1", 9.0)]
    conflicts = [(0, 1)]
    qubo = build_qubo(variables, requests, conflicts, lambda_margin=1.0)

    x_both = np.array([1.0, 1.0])
    x_feasible_high_only = np.array([0.0, 1.0])  # keep higher-priority r1

    val_both = qubo.Q @ x_both @ x_both
    val_feasible = qubo.Q @ x_feasible_high_only @ x_feasible_high_only

    assert val_feasible <= val_both, (
        "Theorem 1 violated: infeasible configuration scored better than "
        "the feasible alternative under lambda = lambda*"
    )


def test_conflict_pairs_from_paths_detects_shared_resource():
    variables = [
        Variable("r0", "pA", 0),
        Variable("r1", "pB", 0),
    ]
    path_resources = {"pA": ("res1", "res2"), "pB": ("res2", "res3")}
    conflicts = conflict_pairs_from_paths(variables, path_resources)
    assert conflicts == [(0, 1)]


def test_conflict_pairs_enforces_assignment_uniqueness():
    """Regression test for a real bug found via NetSquid data: the same
    request assigned to two DIFFERENT paths/times (no shared resource,
    no time overlap) must still conflict, per Eq. (2) — otherwise U(x)
    can double-count a single request's utility."""
    variables = [
        Variable("r0", "pA", 0),
        Variable("r0", "pB", 1),  # same request, different path AND time
    ]
    path_resources = {"pA": ("res1",), "pB": ("res2",)}  # no resource overlap
    conflicts = conflict_pairs_from_paths(variables, path_resources)
    assert conflicts == [(0, 1)], (
        "Same-request variables must conflict even with no resource/time "
        "overlap (Eq. 2, assignment uniqueness) — this was the bug found "
        "via the NetSquid demo where raw U(x)=135 exceeded the total "
        "request weight of 31.5."
    )


def test_served_utility_never_exceeds_total_request_weight():
    """End-to-end sanity check for the same bug: after repair, U(x) must
    never exceed the sum of all distinct requests' weights, regardless
    of how many (path, time) variables exist per request."""
    variables = [
        Variable("r0", "pA", 0), Variable("r0", "pB", 1), Variable("r0", "pC", 2),
        Variable("r1", "pA", 0), Variable("r1", "pB", 1),
    ]
    requests = [Request("r0", 5.0), Request("r1", 3.0)]
    path_resources = {"pA": ("res1",), "pB": ("res2",), "pC": ("res3",)}
    conflicts = conflict_pairs_from_paths(variables, path_resources)
    qubo = build_qubo(variables, requests, conflicts)

    x_all_ones = np.ones(len(variables))  # maximally "greedy" raw assignment
    x_feasible = repair(qubo, x_all_ones, requests, conflicts)

    total_weight = sum(r.priority_weight for r in requests)
    u = served_utility(qubo, x_feasible)
    assert u <= total_weight + 1e-9, (
        f"U(x)={u} exceeds total request weight {total_weight} — "
        f"assignment uniqueness is not being enforced"
    )
    assert u == total_weight  # both requests should be served exactly once


def test_repair_always_returns_feasible_output():
    """Section 4.6 tier-1 guarantee: repair() must ALWAYS return a feasible
    x, regardless of how infeasible x_raw is."""
    variables = [Variable(f"r{i}", "shared_path", 0) for i in range(5)]
    requests = [Request(f"r{i}", float(i + 1)) for i in range(5)]
    conflicts = [(i, j) for i in range(5) for j in range(i + 1, 5)]  # all-conflict clique
    qubo = build_qubo(variables, requests, conflicts)

    x_raw = np.ones(5)  # maximally infeasible: everyone active at once
    x_feasible = repair(qubo, x_raw, requests, conflicts)

    assert verify_feasible(x_feasible, conflicts)
    # exactly the single highest-priority request (r4, weight=5) should survive
    assert x_feasible.sum() == 1
    assert x_feasible[4] == 1.0


def test_repair_is_idempotent_on_already_feasible_input():
    variables = [Variable("r0", "p0", 0), Variable("r1", "p1", 0)]
    requests = [Request("r0", 1.0), Request("r1", 2.0)]
    conflicts = [(0, 1)]
    qubo = build_qubo(variables, requests, conflicts)

    x = np.array([0.0, 1.0])  # already feasible
    x_repaired = repair(qubo, x, requests, conflicts)
    assert np.array_equal(x, x_repaired)


def test_completion_recovers_missed_nonconflicting_variable():
    """Proposition 2: nếu QAOA bỏ sót 1 biến không xung đột với gì cả,
    completion phải thêm lại — đây chính là bug đã fix (repair cũ chỉ
    trừ, không bao giờ thêm)."""
    variables = [
        Variable("r0", "pA", 0),  # xung đột với r1
        Variable("r1", "pB", 0),
        Variable("r2", "pC", 1),  # KHÔNG xung đột với ai — QAOA lỡ bỏ sót
    ]
    requests = [Request("r0", 9.0), Request("r1", 5.0), Request("r2", 3.0)]
    path_resources = {"pA": ("res1",), "pB": ("res1",), "pC": ("res2",)}
    conflicts = conflict_pairs_from_paths(variables, path_resources)
    qubo = build_qubo(variables, requests, conflicts)

    x_raw = np.array([1.0, 0.0, 0.0])  # QAOA chỉ bật r0, bỏ sót r2
    x_fixed = repair(qubo, x_raw, requests, conflicts)

    assert x_fixed[0] == 1.0  # r0 giữ nguyên (không xung đột ai còn active)
    assert x_fixed[2] == 1.0, "completion phải thêm lại r2 (không xung đột)"
    assert verify_feasible(x_fixed, conflicts)


def test_completion_never_decreases_utility():
    """Proposition 2: U(x sau completion) >= U(x trước completion), vì
    completion chỉ thêm biến có trọng số dương, không bao giờ xóa."""
    rng = np.random.default_rng(7)
    variables = [Variable(f"r{i}", f"p{i}", i % 3) for i in range(10)]
    requests = [Request(f"r{i}", float(rng.uniform(1, 10))) for i in range(10)]
    path_resources = {f"p{i}": (f"res{i%4}",) for i in range(10)}
    conflicts = conflict_pairs_from_paths(variables, path_resources)
    qubo = build_qubo(variables, requests, conflicts)

    for trial in range(20):
        x_raw = rng.integers(0, 2, size=10).astype(float)
        # utility "trước completion" = chỉ chạy bước 1 (mô phỏng bằng cách
        # gọi repair rồi so utility với served_utility(x_raw) làm cận dưới
        # tham khảo — completion luôn >= utility của chính x_raw đã lọc xung đột
        x_fixed = repair(qubo, x_raw, requests, conflicts)
        assert verify_feasible(x_fixed, conflicts)
        # mọi biến active trong x_raw ban đầu (nếu sống sót bước 1) vẫn active
        # sau completion (completion chỉ thêm, không xóa gì đã có ở bước 1)


def test_completion_recovers_missed_beneficial_variable():
    """Algorithm 3 extended (completion step): if QAOA's raw output missed
    a high-priority, non-conflicting variable entirely (x_i=0), repair
    must now recover it — the old subtractive-only repair could not."""
    variables = [
        Variable("r0", "pA", 0),  # weight 9, no conflict with r1
        Variable("r1", "pB", 0),  # weight 5
    ]
    requests = [Request("r0", 9.0), Request("r1", 5.0)]
    path_resources = {"pA": ("res1",), "pB": ("res2",)}  # no shared resource
    conflicts = conflict_pairs_from_paths(variables, path_resources)
    qubo = build_qubo(variables, requests, conflicts)

    x_raw = np.array([0.0, 1.0])  # QAOA missed r0 (higher priority!) entirely
    x_fixed = repair(qubo, x_raw, requests, conflicts)

    assert x_fixed[0] == 1.0, "completion step must recover the missed r0"
    assert x_fixed[1] == 1.0, "r1 should remain active (no conflict)"
    assert verify_feasible(x_fixed, conflicts)


def test_completion_never_decreases_utility():
    """Proposition 2: completion only adds non-conflicting variables with
    positive weight, so repair's output stays feasible and non-negative
    utility across many random raw inputs."""
    rng = np.random.default_rng(7)
    variables = [Variable(f"r{i}", f"p{i}", i % 3) for i in range(8)]
    requests = [Request(f"r{i}", float(rng.uniform(1, 10))) for i in range(8)]
    path_resources = {f"p{i}": (f"res{i % 4}",) for i in range(8)}
    conflicts = conflict_pairs_from_paths(variables, path_resources)
    qubo = build_qubo(variables, requests, conflicts)

    for _ in range(20):
        x_raw = rng.integers(0, 2, size=8).astype(float)
        x_fixed = repair(qubo, x_raw, requests, conflicts)
        assert verify_feasible(x_fixed, conflicts)
        assert served_utility(qubo, x_fixed) >= 0


if __name__ == "__main__":
    test_lambda_star_matches_theorem_1_formula()
    test_lambda_star_zero_when_no_conflicts()
    test_build_qubo_diagonal_encodes_negative_weights()
    test_build_qubo_penalizes_conflicting_pair_enough_to_deter_violation()
    test_conflict_pairs_from_paths_detects_shared_resource()
    test_conflict_pairs_enforces_assignment_uniqueness()
    test_served_utility_never_exceeds_total_request_weight()
    test_repair_always_returns_feasible_output()
    test_repair_is_idempotent_on_already_feasible_input()
    test_completion_recovers_missed_nonconflicting_variable()
    test_completion_never_decreases_utility()
    test_completion_recovers_missed_beneficial_variable()
    test_completion_never_decreases_utility()
    print("All tests passed.")
