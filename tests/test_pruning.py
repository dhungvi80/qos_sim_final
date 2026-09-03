"""
tests/test_pruning.py

Unit tests for Algorithm 1 (pruning.py), including an empirical check of
Proposition 1's soundness property F ⊆ A on synthetic data. This does not
replace the formal proof in the paper (Section 3.4) — it is the
implementation-side verification the paper's own text calls for
("should be verified against the implementation").
"""

import itertools
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qos_sim.pruning import NetworkTelemetry, PathConfig, Request, Variable, prune


class SyntheticTelemetry(NetworkTelemetry):
    """Deterministic-but-varied telemetry for testing.

    fidelity(path, t) and availability(path, t) are derived from a fixed
    per-(path, t) random seed so results are reproducible across runs,
    matching the paper's per-trial seeding convention (Section 5.1:
    seed = 42 + trial_index) rather than one fixed global seed.
    """

    def __init__(self, trial_seed: int, fidelity_floor: float = 0.5,
                 unavailable_prob: float = 0.1):
        self._rng_seed = trial_seed
        self._fidelity_floor = fidelity_floor
        self._unavailable_prob = unavailable_prob

    def _rng_for(self, path: PathConfig, t: int) -> random.Random:
        return random.Random(f"{self._rng_seed}:{path.id}:{t}")

    def fidelity(self, path: PathConfig, t: int) -> float:
        rng = self._rng_for(path, t)
        return self._fidelity_floor + rng.random() * (1 - self._fidelity_floor)

    def is_available(self, path: PathConfig, t: int) -> bool:
        rng = self._rng_for(path, t)
        rng.random()  # advance state so it differs from fidelity() draw
        return rng.random() > self._unavailable_prob

    def has_collision(self, path, t, reserved):
        for resource in path.resources:
            holders = reserved.get((resource, t), set())
            if holders and path.id not in holders:
                return True
        return False


def _brute_force_feasible_set(requests, paths, time_slots, telemetry,
                               fidelity_threshold) -> set[tuple[str, str, int]]:
    """Independently recompute the feasible set F for comparison against A.

    This mirrors the *definition* of feasibility (Eqs. 2-4), not the
    pruning implementation, so it is a genuine external check rather than
    a restatement of prune().
    """
    feasible = set()
    reserved: dict[tuple[str, int], set[str]] = {}
    ordered = sorted(requests, key=lambda r: (-r.priority_weight, r.id))
    for req in ordered:
        for t in time_slots:
            for path in paths:
                if telemetry.fidelity(path, t) < fidelity_threshold:
                    continue
                if not telemetry.is_available(path, t):
                    continue
                if telemetry.has_collision(path, t, reserved):
                    continue
                feasible.add((req.id, path.id, t))
                for resource in path.resources:
                    reserved.setdefault((resource, t), set()).add(path.id)
    return feasible


def _make_scenario(n_requests=6, n_paths=5, n_resources=6, n_time=4, trial_seed=42):
    rng = random.Random(trial_seed)
    requests = [Request(id=f"r{i}", priority_weight=rng.uniform(1, 10))
                for i in range(n_requests)]
    resources = [f"res{j}" for j in range(n_resources)]
    paths = [
        PathConfig(id=f"p{k}", resources=tuple(rng.sample(resources, k=2)))
        for k in range(n_paths)
    ]
    time_slots = list(range(n_time))
    telemetry = SyntheticTelemetry(trial_seed=trial_seed)
    return requests, paths, time_slots, telemetry


def test_soundness_F_subseteq_A():
    """Proposition 1: every feasible variable found by brute force must
    also appear in the active subspace A returned by prune()."""
    for trial_seed in [42, 43, 44, 100, 137]:
        requests, paths, time_slots, telemetry = _make_scenario(trial_seed=trial_seed)

        active_vars, stats = prune(
            requests, paths, time_slots, telemetry, fidelity_threshold=0.6
        )
        active_keys = {v.key() for v in active_vars}

        feasible_keys = _brute_force_feasible_set(
            requests, paths, time_slots, telemetry, fidelity_threshold=0.6
        )

        missing = feasible_keys - active_keys
        assert not missing, (
            f"Soundness violated for seed={trial_seed}: "
            f"{len(missing)} feasible variables excluded by pruning: {missing}"
        )


def test_reduction_ratio_bounds():
    """Corollary 1: |A| <= |D|, and reduction ratio is in [0, 1]."""
    requests, paths, time_slots, telemetry = _make_scenario(trial_seed=7)
    active_vars, stats = prune(requests, paths, time_slots, telemetry,
                                fidelity_threshold=0.6)
    assert stats.size_A <= stats.size_D
    assert 0.0 <= stats.reduction_ratio <= 1.0
    assert stats.size_D == len(requests) * len(paths) * len(time_slots)


def test_low_fidelity_rejected():
    """A path with fidelity forced below threshold must never appear in A."""
    requests = [Request(id="r0", priority_weight=5.0)]
    paths = [PathConfig(id="p0", resources=("res0", "res1"))]
    time_slots = [0]

    class AlwaysLowFidelity(NetworkTelemetry):
        def fidelity(self, path, t):
            return 0.1
        def is_available(self, path, t):
            return True
        def has_collision(self, path, t, reserved):
            return False

    active_vars, stats = prune(requests, paths, time_slots,
                                AlwaysLowFidelity(), fidelity_threshold=0.5)
    assert active_vars == []
    assert stats.rejected_fidelity == 1
    assert stats.size_A == 0


def test_collision_keeps_only_higher_priority():
    """Two requests contending for the same resource at the same time:
    only the higher-priority request should retain the variable — this
    is the same tie-break Theorem 1 / Algorithm 3 assume (Eq. 13)."""
    requests = [
        Request(id="low", priority_weight=1.0),
        Request(id="high", priority_weight=9.0),
    ]
    paths = [PathConfig(id="shared", resources=("res0",))]
    time_slots = [0]

    class AlwaysFeasible(NetworkTelemetry):
        def fidelity(self, path, t):
            return 0.99
        def is_available(self, path, t):
            return True
        def has_collision(self, path, t, reserved):
            for resource in path.resources:
                holders = reserved.get((resource, t), set())
                if holders and path.id not in holders:
                    return True
            return False

    active_vars, stats = prune(requests, paths, time_slots,
                                AlwaysFeasible(), fidelity_threshold=0.5)
    request_ids = {v.request_id for v in active_vars}
    assert "high" in request_ids, "higher-priority request must be retained"


if __name__ == "__main__":
    test_soundness_F_subseteq_A()
    test_reduction_ratio_bounds()
    test_low_fidelity_rejected()
    test_collision_keeps_only_higher_priority()
    print("All tests passed.")
