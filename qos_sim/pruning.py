"""
pruning.py — Algorithm 1 (Real-time Variable Pruning)

Reference: "A Quantum Optimization Approach for Dynamic Resource Allocation
in Quantum Networks", Section 3.3 (Algorithm 1) and Section 3.4
(Proposition 1 / Corollary 1).

IMPORTANT — do not modify the feasibility predicate without also updating
Proposition 1 in the paper. Proposition 1's proof relies on the pruning
operator using *exactly* the three conditions below (fidelity, temporal
availability, no collision) and nothing else. Adding any additional
heuristic filter here invalidates the soundness guarantee F ⊆ A stated
in Eq. (9)-(10) of the paper, and Theorem 1 / Algorithm 3 (which assume
A was built this way) would need to be re-derived.

Notation follows the paper:
    R       — request set
    K       — candidate path/route set
    T       — discrete time slots
    D       — full decision space, |D| = |R| * |K| * |T|
    A       — active (pruned) subspace returned by Algorithm 1
    F_min   — minimum acceptable fidelity threshold
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterable, Mapping, Optional


# --------------------------------------------------------------------------- #
# Data model
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class Request:
    """An entanglement request i in R."""
    id: str
    priority_weight: float  # w_i, used later by qubo_builder.py / Theorem 1


@dataclass(frozen=True)
class PathConfig:
    """A candidate path/route k in K.

    resources: the ordered set of physical resources (links/nodes) this
    path occupies. Used for the collision (mutual-exclusivity) test.
    """
    id: str
    resources: tuple[str, ...]


@dataclass(frozen=True)
class Variable:
    """A single decision-variable triple (i, k, t) in D."""
    request_id: str
    path_id: str
    time_slot: int

    def key(self) -> tuple[str, str, int]:
        return (self.request_id, self.path_id, self.time_slot)


class NetworkTelemetry:
    """Live simulator telemetry queried by the feasibility test T(i,k,t).

    This is an interface, not an implementation: `simulator.py`
    (NetSquid-backed) must provide an object satisfying this protocol.
    Kept separate so pruning.py has no NetSquid dependency and can be
    unit-tested with synthetic telemetry (see tests/test_pruning.py).
    """

    def fidelity(self, path: PathConfig, t: int) -> float:
        """Estimated end-to-end fidelity of `path` at time slot `t`."""
        raise NotImplementedError

    def is_available(self, path: PathConfig, t: int) -> bool:
        """Whether all resources on `path` are physically available
        (e.g. not mid-generation, not already reserved by a prior
        higher-priority allocation) at time slot `t`."""
        raise NotImplementedError

    def has_collision(self, path: PathConfig, t: int,
                       reserved: Mapping[tuple[str, int], set[str]]) -> bool:
        """Whether `path` at time `t` collides with resources already
        reserved by variables retained earlier in this pruning pass.

        `reserved` maps (resource_id, time_slot) -> set of path_ids
        currently holding that resource at that time slot.
        """
        raise NotImplementedError


@dataclass
class PruningStats:
    """Diagnostics for Corollary 1 (Expected Reduction Ratio, Eq. 11-12).

    Log these per scenario so retention probability p and the reduction
    ratio |A|/|D| can be reported and checked against the Hoeffding bound
    in the paper, rather than only reporting the final count.
    """
    size_D: int = 0
    size_A: int = 0
    rejected_fidelity: int = 0
    rejected_availability: int = 0
    rejected_collision: int = 0

    @property
    def reduction_ratio(self) -> float:
        """|A| / |D|  — empirical estimate of p in Corollary 1."""
        return self.size_A / self.size_D if self.size_D else 0.0

    def as_dict(self) -> dict:
        return {
            "size_D": self.size_D,
            "size_A": self.size_A,
            "reduction_ratio": self.reduction_ratio,
            "rejected_fidelity": self.rejected_fidelity,
            "rejected_availability": self.rejected_availability,
            "rejected_collision": self.rejected_collision,
        }


# --------------------------------------------------------------------------- #
# Algorithm 1
# --------------------------------------------------------------------------- #

def prune(
    requests: Iterable[Request],
    paths: Iterable[PathConfig],
    time_slots: Iterable[int],
    telemetry: NetworkTelemetry,
    fidelity_threshold: float,
    priority_order: Optional[Callable[[Request], float]] = None,
    valid_paths: Optional[dict[str, set[str]]] = None,
) -> tuple[list[Variable], PruningStats]:
    """Algorithm 1: Real-time Variable Pruning.

    Implements, in order, the six steps of Algorithm 1 (paper Section 3.3):
        1. Initialize the active set.
        2. For each (i,k,t), retrieve current fidelity and availability.
        3. Test whether the path satisfies feasibility conditions.
        4. Retain variables that satisfy fidelity, timing, and collision
           constraints.
        5. Remove all variables that violate any hard constraint.
        6. Return the reduced active subspace.

    The three feasibility conditions tested (step 3) are EXACTLY the
    predicate T(i,k,t) used in Proposition 1:
        T(i,k,t) = [fidelity >= F_min] AND [temporal availability]
                   AND [no resource collision]
    Do not add a fourth condition here — see module docstring.

    valid_paths : optional map request_id -> set of path_ids that
        actually connect that request's source-destination pair. On a
        real topology, most (request, path) pairs are not physically
        meaningful (a path between two unrelated nodes cannot serve a
        request between a different pair) — this was missing in earlier
        small demo scenarios where every path happened to be usable by
        every request, masking the gap. If None (default), every path is
        considered valid for every request, preserving old behavior for
        small synthetic tests. When provided, |D| = sum over requests of
        |valid_paths[req.id]| * |time_slots| — NOT
        |requests|*|paths|*|time_slots| — since the latter would count
        physically meaningless (request, path) combinations that were
        never real candidates in the first place.

    Requests are processed in descending priority_weight order (ties
    broken by id) so that, when two variables would otherwise collide,
    the higher-priority request is the one already holding the resource
    when the lower-priority one is tested — consistent with the
    tie-breaking rule used later in Algorithm 3 / Theorem 1 (Eq. 13).

    Returns
    -------
    active_variables : list[Variable]
        The active subspace A.
    stats : PruningStats
        Counts needed to log p = |A|/|D| for Corollary 1 (Eq. 11-12).
    """
    requests = list(requests)
    paths = list(paths)
    time_slots = list(time_slots)
    paths_by_id = {p.id: p for p in paths}

    key_fn = priority_order or (lambda r: r.priority_weight)
    ordered_requests = sorted(requests, key=lambda r: (-key_fn(r), r.id))

    if valid_paths is None:
        size_D = len(requests) * len(paths) * len(time_slots)
    else:
        size_D = sum(len(valid_paths.get(r.id, ())) for r in requests) * len(time_slots)
    stats = PruningStats(size_D=size_D)

    # reserved[(resource_id, t)] = set of path_ids currently holding it
    reserved: dict[tuple[str, int], set[str]] = {}
    active_variables: list[Variable] = []

    # Step 1: initialize the active set (done above: active_variables = [])
    for req in ordered_requests:
        candidate_paths = (
            paths if valid_paths is None
            else [paths_by_id[pid] for pid in valid_paths.get(req.id, ()) if pid in paths_by_id]
        )
        for t in time_slots:
            for path in candidate_paths:
                # Step 2: retrieve current fidelity and availability
                fid = telemetry.fidelity(path, t)
                available = telemetry.is_available(path, t)

                # Step 3: test feasibility conditions
                if fid < fidelity_threshold:
                    stats.rejected_fidelity += 1
                    continue
                if not available:
                    stats.rejected_availability += 1
                    continue
                if telemetry.has_collision(path, t, reserved):
                    stats.rejected_collision += 1
                    continue

                # Step 4: retain — variable passes all three conditions
                active_variables.append(Variable(req.id, path.id, t))
                for resource in path.resources:
                    reserved.setdefault((resource, t), set()).add(path.id)

                # Step 5 (implicit): anything not appended here is removed
                # by construction — no separate pass is needed since we
                # only ever append variables that pass step 3.

    stats.size_A = len(active_variables)
    # Step 6: return the reduced active subspace
    return active_variables, stats
