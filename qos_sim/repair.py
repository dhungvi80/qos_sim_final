"""
repair.py — Algorithm 3 (Constraint Repair and Feasibility Projection),
Section 4.4 — EXTENDED with a completion (top-up) pass.

Original Algorithm 3 only resolved conflicts within x_raw's active support
(subtractive-only: could drop variables, never add missed ones). Empirical
100-trial data showed this caps achievable utility regardless of QAOA
search quality, since any request QAOA failed to activate could never be
recovered. This version adds Step 3 (completion): after conflict
resolution, greedily activate any still-unserved, non-conflicting request
from the remainder of the active subspace A — never removes anything from
Step 1's output, only adds. See Proposition 2 (paper) for why this cannot
decrease utility.

The Step 1 tie-break rule is unchanged and still matches the Delta_c
comparison of Theorem 1 (Eq. 14): retaining the higher-priority request
dominates any single-constraint violation whenever lambda >= lambda*
(Eq. 15). Theorem 1 itself concerns the QUBO's global-minimizer structure
and does not depend on Algorithm 3's completeness, so it requires no
restatement — only the repair heuristic's own guarantee changes.
"""

from __future__ import annotations

import numpy as np

from qos_sim.pruning import Request, Variable
from qos_sim.qubo_builder import QUBOInstance


def repair(
    qubo: QUBOInstance,
    x_raw: np.ndarray,
    requests: list[Request],
    conflicts: list[tuple[int, int]],
) -> np.ndarray:
    """Algorithm 3 (extended): Constraint Repair and Feasibility Projection
    with Completion.

    Input: raw measured bitstring x_raw (0/1 array, one entry per variable
        in qubo.variables), from Algorithm 2 (qaoa_numpy.sample_bitstring).
    Output: x_feasible — a 0/1 array satisfying all exclusivity constraints
        among `conflicts`, weakly dominating (Proposition 2) the original
        (completion-free) repaired solution in total served utility.

    Steps:
        1. If two assignments conflict, retain the one with higher request
           priority (Theorem 1 / Eq. 14 tie-break). [unchanged]
        2. (Fidelity/timing constraints are already enforced by pruning.py
           before this point.)
        3. NEW — Completion: consider every variable NOT active after
           step 1, in descending priority-weight order; activate it if
           doing so introduces no conflict with the currently active set.
           This recovers requests that Algorithm 2's variational search
           failed to activate, which step 1 alone could never restore.
    """
    x = x_raw.copy().astype(float)
    weight_of = {r.id: r.priority_weight for r in requests}

    # Step 1: resolve every conflicting pair by keeping the
    # higher-priority request's variable active.
    for a, b in conflicts:
        if x[a] > 0.5 and x[b] > 0.5:
            wa = weight_of[qubo.variables[a].request_id]
            wb = weight_of[qubo.variables[b].request_id]
            if wa >= wb:
                x[b] = 0.0
            else:
                x[a] = 0.0

    # Step 3: completion pass — build adjacency once, then greedily add
    # any currently-inactive variable that does not conflict with the
    # active set, processing candidates by descending priority weight.
    n = len(qubo.variables)
    adjacency: dict[int, set[int]] = {i: set() for i in range(n)}
    for a, b in conflicts:
        adjacency[a].add(b)
        adjacency[b].add(a)

    active = {i for i in range(n) if x[i] > 0.5}
    inactive_order = sorted(
        (i for i in range(n) if x[i] <= 0.5),
        key=lambda i: -weight_of[qubo.variables[i].request_id],
    )
    for i in inactive_order:
        if not (adjacency[i] & active):
            x[i] = 1.0
            active.add(i)

    return x


def verify_feasible(x: np.ndarray, conflicts: list[tuple[int, int]]) -> bool:
    """Sanity check used by tests / logging: True iff no conflicting pair
    is simultaneously active in x. This should always be True for any
    output of `repair()` — use it to log feasibility_rate for Table 6."""
    return all(not (x[a] > 0.5 and x[b] > 0.5) for a, b in conflicts)
