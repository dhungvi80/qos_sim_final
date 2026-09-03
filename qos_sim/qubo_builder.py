"""
qubo_builder.py — Section 3.5 (QUBO Reformulation) and Section 3.6
(Theorem 1: Tight Network-Specific Penalty Bound).

Takes the active subspace A returned by pruning.py (Algorithm 1) and builds
the QUBO matrix Q such that

    min x^T Q x      (Eq. 13)

is equivalent to the original constrained allocation problem restricted to A
(Proposition 1 guarantees this restriction loses no optimal solutions).

IMPORTANT — keep in sync with the paper:
    lambda_star() implements Theorem 1 (Eq. 13-15) EXACTLY as derived in the
    paper: lambda* = max over conflicting pairs (i,j) of min(w_i, w_j).
    Do not substitute a generic "large enough" heuristic here — the whole
    point of Theorem 1 is that lambda* is computed, not guessed.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np

from qos_sim.pruning import Request, Variable


@dataclass
class QUBOInstance:
    """A built QUBO problem instance.

    Q : (n, n) numpy array — the QUBO matrix such that min x^T Q x
        (Eq. 13) is minimized over x in {0,1}^n.
    index_of : maps a Variable's key (request_id, path_id, time_slot) to
        its position in x, so a solution bitstring can be decoded back into
        (request, path, time) assignments.
    variables : the active subspace A, in the same order as Q's rows/cols.
    lambda_used : the penalty coefficient actually used to build Q.
    lambda_star : the Theorem 1 bound (Eq. 15); lambda_used must be >= this
        for the feasibility guarantee to hold.
    """
    Q: np.ndarray
    index_of: dict[tuple[str, str, int], int]
    variables: list[Variable]
    lambda_used: float
    lambda_star: float


def conflict_pairs_from_paths(
    variables: list[Variable],
    path_resources: dict[str, tuple[str, ...]],
) -> list[tuple[int, int]]:
    """Proper conflict-pair extraction covering BOTH hard constraints
    from Eqs. (2)-(3):

    (a) Assignment uniqueness (Eq. 2): each request i may be served by at
        most ONE (path, time) combination. Two variables sharing the same
        request_id conflict regardless of path/time — this was MISSING
        from earlier versions of this function, which only checked
        resource overlap. Without it, the objective can double-count a
        single request's utility across multiple simultaneous
        assignments, inflating U(x) beyond the true maximum (sum of all
        distinct requests' weights) — caught via a real NetSquid run
        where raw U(x) exceeded the total available weight.

    (b) Resource exclusivity (Eq. 3): two variables for DIFFERENT
        requests conflict if they occupy the same time slot and their
        paths share at least one physical resource.
    """
    pairs = set()

    # (a) assignment uniqueness: any two variables for the same request
    by_request: dict[str, list[int]] = {}
    for idx, v in enumerate(variables):
        by_request.setdefault(v.request_id, []).append(idx)
    for idxs in by_request.values():
        for a, b in combinations(idxs, 2):
            pairs.add((a, b))

    # (b) resource exclusivity: same time slot, different requests, shared resource
    by_time: dict[int, list[int]] = {}
    for idx, v in enumerate(variables):
        by_time.setdefault(v.time_slot, []).append(idx)
    for idxs in by_time.values():
        for a, b in combinations(idxs, 2):
            va, vb = variables[a], variables[b]
            if va.request_id == vb.request_id:
                continue  # already covered by (a)
            ra = set(path_resources.get(va.path_id, ()))
            rb = set(path_resources.get(vb.path_id, ()))
            if ra & rb:
                pairs.add((a, b))

    return sorted(pairs)


def lambda_star(
    variables: list[Variable],
    weight_of: dict[str, float],
    conflicts: list[tuple[int, int]],
) -> float:
    """Theorem 1 (Eq. 13-15): the tight, network-specific penalty bound.

        lambda* = max_{(i,j) in conflicts} min(w_i, w_j)

    This is the exact quantity from the paper's proof: for a conflict pair
    (i, j), the maximum utility gain from violating exclusivity is
    Delta_c = w_i + w_j - max(w_i, w_j) = min(w_i, w_j) (Eq. 13). Taking
    lambda >= max over all conflicts of Delta_c dominates every possible
    single-constraint violation (Eq. 15).

    Returns 0.0 if there are no conflicts (degenerate case — any lambda >= 0
    trivially satisfies Theorem 1).
    """
    if not conflicts:
        return 0.0
    gains = []
    for a, b in conflicts:
        wa = weight_of[variables[a].request_id]
        wb = weight_of[variables[b].request_id]
        gains.append(min(wa, wb))
    return max(gains)


def build_qubo(
    variables: list[Variable],
    requests: list[Request],
    conflicts: list[tuple[int, int]],
    lambda_margin: float = 1.0,
) -> QUBOInstance:
    """Build the QUBO matrix Q (Eq. 13) over the active subspace.

    Diagonal terms encode the (negated, since we minimize) utility reward
    -w_i for each variable. Off-diagonal terms encode the penalty
    lambda * x_a * x_b for each conflicting pair (a, b), so that setting
    both x_a = x_b = 1 costs `lambda` in addition to their utility, per
    Theorem 1's proof.

    lambda_margin >= 1.0 scales lambda_star up slightly (default: no
    margin, i.e. exactly at the Theorem 1 bound). A small margin (e.g. 1.05)
    is a reasonable numerical-safety choice in practice, but note this
    changes lambda_used away from the exact lambda_star reported in the
    paper — log both separately (see QUBOInstance).
    """
    n = len(variables)
    index_of = {v.key(): i for i, v in enumerate(variables)}
    weight_of = {r.id: r.priority_weight for r in requests}

    lam_star = lambda_star(variables, weight_of, conflicts)
    lam_used = lam_star * lambda_margin

    Q = np.zeros((n, n))
    for i, v in enumerate(variables):
        Q[i, i] -= weight_of[v.request_id]  # -w_i on the diagonal (Eq. 1 term)

    for a, b in conflicts:
        # split lambda across the symmetric off-diagonal pair so that
        # x^T Q x contributes lambda_used * x_a * x_b exactly once
        Q[a, b] += lam_used / 2.0
        Q[b, a] += lam_used / 2.0

    return QUBOInstance(
        Q=Q, index_of=index_of, variables=variables,
        lambda_used=lam_used, lambda_star=lam_star,
    )


def objective_value(qubo: QUBOInstance, x: np.ndarray) -> float:
    """Raw QUBO objective x^T Q x (Eq. 13) — NOT the same as the utility
    U(x) of Eq. (1) once penalty terms are included; use `served_utility`
    below to report the paper's actual utility metric for Table 6/7/8."""
    return float(x @ qubo.Q @ x)


def served_utility(qubo: QUBOInstance, x: np.ndarray) -> float:
    """Sum of w_i over variables set to 1 in x — the utility metric U(x)
    of Eq. (1), independent of the penalty term. This is what Table 6's
    'Total Served Priority' column should report, not the raw QUBO value."""
    total = 0.0
    weight_of = {v.request_id: None for v in qubo.variables}
    # recover weights from the diagonal (which stores -w_i)
    for i, v in enumerate(qubo.variables):
        if x[i] > 0.5:
            total += -qubo.Q[i, i]
    return total
