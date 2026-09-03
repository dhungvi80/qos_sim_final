"""
examples/demo_pruning.py

Minimal runnable example of Algorithm 1, using synthetic telemetry
(no NetSquid dependency yet — simulator.py will supply a real
NetworkTelemetry implementation in the next stage).

Run:
    python examples/demo_pruning.py
"""

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qos_sim.pruning import NetworkTelemetry, PathConfig, Request, prune


class DemoTelemetry(NetworkTelemetry):
    """Stand-in telemetry: fidelity decays with time slot, ~15% of
    (path, t) pairs are randomly unavailable. Replace with the real
    NetSquid-backed implementation once simulator.py exists."""

    def __init__(self, seed: int):
        self._rng = random.Random(seed)
        self._base_fidelity = {}

    def fidelity(self, path: PathConfig, t: int) -> float:
        key = (path.id, t)
        if key not in self._base_fidelity:
            self._base_fidelity[key] = self._rng.uniform(0.55, 0.99)
        return self._base_fidelity[key] * (0.97 ** t)  # simple decay model

    def is_available(self, path, t):
        return self._rng.random() > 0.15

    def has_collision(self, path, t, reserved):
        for resource in path.resources:
            holders = reserved.get((resource, t), set())
            if holders and path.id not in holders:
                return True
        return False


def build_25_node_scenario(trial_seed: int):
    """Small stand-in for the 25-node scenario in Table 8 — not the real
    topology yet, just enough structure to exercise the pruning logic
    end-to-end before simulator.py exists."""
    rng = random.Random(trial_seed)
    requests = [Request(id=f"req{i}", priority_weight=rng.uniform(1, 10))
                for i in range(25)]
    resources = [f"link{j}" for j in range(30)]
    paths = [PathConfig(id=f"path{k}", resources=tuple(rng.sample(resources, k=3)))
              for k in range(15)]
    time_slots = list(range(8))
    return requests, paths, time_slots


def main():
    trial_seed = 42  # paper convention: seed = 42 + trial_index (Section 5.1)
    requests, paths, time_slots = build_25_node_scenario(trial_seed)
    telemetry = DemoTelemetry(seed=trial_seed)

    active_vars, stats = prune(
        requests, paths, time_slots, telemetry, fidelity_threshold=0.6
    )

    print(f"|D| = {stats.size_D}")
    print(f"|A| = {stats.size_A}")
    print(f"reduction ratio |A|/|D| = {stats.reduction_ratio:.4f}  "
          f"(compare against Corollary 1 / Table 8's ~0.22 for 25 nodes)")
    print(f"rejected — fidelity: {stats.rejected_fidelity}, "
          f"availability: {stats.rejected_availability}, "
          f"collision: {stats.rejected_collision}")


if __name__ == "__main__":
    main()
