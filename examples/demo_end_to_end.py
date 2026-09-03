"""
examples/demo_end_to_end.py — Algorithm 4 (End-to-End Framework Execution),
Section 4.5.1, run against a small synthetic scenario.

Pipeline: Algorithm 1 (pruning.py) -> QUBO + lambda* (qubo_builder.py) ->
GWO-driven QAOA search (gwo_optimizer.py + qaoa_numpy.py) -> Algorithm 3
(repair.py) -> feasible schedule with its objective value U(x) (Eq. 1).

Kept to <= 10 active qubits so the numpy statevector simulator (a stand-in
for the Qiskit-backed DVQE kernel, see qaoa_numpy.py docstring) runs in
well under a second. Swap in real Qiskit Aer + NetSquid telemetry on the
research server without changing this script's structure.

Run:
    python examples/demo_end_to_end.py
"""

import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qos_sim.pruning import NetworkTelemetry, PathConfig, Request, prune
from qos_sim.qubo_builder import build_qubo, conflict_pairs_from_paths, objective_value, served_utility
from qos_sim.qaoa_numpy import run_qaoa
from qos_sim.gwo_optimizer import optimize as gwo_optimize
from qos_sim.repair import repair, verify_feasible


class DemoTelemetry(NetworkTelemetry):
    def __init__(self, seed: int):
        self._rng = random.Random(seed)
        self._fid_cache = {}

    def fidelity(self, path, t):
        key = (path.id, t)
        if key not in self._fid_cache:
            self._fid_cache[key] = self._rng.uniform(0.5, 0.99)
        return self._fid_cache[key]

    def is_available(self, path, t):
        return self._rng.random() > 0.1

    def has_collision(self, path, t, reserved):
        for resource in path.resources:
            holders = reserved.get((resource, t), set())
            if holders and path.id not in holders:
                return True
        return False


def build_small_scenario(trial_seed: int):
    """A deliberately small scenario (aiming for <=10 active variables
    after pruning) so the demo runs end-to-end without needing DVQE's
    Hamiltonian partitioning (Section 4.3.1) — real 25/50/100-node
    scenarios must go through the partitioned DVQE path instead."""
    rng = random.Random(trial_seed)
    requests = [Request(id=f"req{i}", priority_weight=rng.uniform(1, 10))
                for i in range(4)]
    resources = [f"link{j}" for j in range(4)]
    paths = [PathConfig(id=f"path{k}", resources=tuple(rng.sample(resources, k=2)))
              for k in range(3)]
    time_slots = list(range(2))
    return requests, paths, time_slots


def main():
    trial_seed = 42  # paper convention: seed = 42 + trial_index (Section 5.1)
    requests, paths, time_slots = build_small_scenario(trial_seed)
    telemetry = DemoTelemetry(seed=trial_seed)
    path_resources = {p.id: p.resources for p in paths}

    # --- Algorithm 1: Pruning ---
    active_vars, prune_stats = prune(
        requests, paths, time_slots, telemetry, fidelity_threshold=0.55
    )
    print(f"[Algorithm 1] |D|={prune_stats.size_D}  |A|={prune_stats.size_A}  "
          f"reduction ratio={prune_stats.reduction_ratio:.3f}")

    if len(active_vars) == 0:
        print("Active subspace is empty for this seed/threshold — try a different seed.")
        return
    if len(active_vars) > 12:
        print(f"Active subspace has {len(active_vars)} variables; trimming to 10 "
              f"for the numpy statevector demo (real runs would invoke DVQE "
              f"partitioning here instead, Section 4.3.1).")
        active_vars = active_vars[:10]

    # --- Section 3.5-3.6: QUBO + Theorem 1's lambda* ---
    conflicts = conflict_pairs_from_paths(active_vars, path_resources)
    qubo = build_qubo(active_vars, requests, conflicts)
    print(f"[Theorem 1] lambda* = {qubo.lambda_star:.4f}  "
          f"(lambda_used = {qubo.lambda_used:.4f})")
    print(f"n_qubits = {len(active_vars)}, conflict pairs = {len(conflicts)}")

    # --- Algorithm 2 (numpy stand-in) + Algorithm 4's GWO outer loop ---
    p_depth = 2

    def fitness(params: np.ndarray) -> float:
        gamma, beta = params[:p_depth], params[p_depth:]
        # single-layer-averaged expectation for this simple demo optimizer
        result = run_qaoa(qubo.Q, gamma.mean(), beta.mean(), p=p_depth)
        return result.expectation

    rng = np.random.default_rng(trial_seed)
    gwo_result = gwo_optimize(fitness, dim=2 * p_depth, n_wolves=15,
                               max_iter=50, rng=rng)
    print(f"[GWO] best <H_C> found = {gwo_result.best_fitness:.4f} "
          f"after {len(gwo_result.history)} iterations")

    gamma_best = gwo_result.best_position[:p_depth].mean()
    beta_best = gwo_result.best_position[p_depth:].mean()
    qaoa_result = run_qaoa(qubo.Q, gamma_best, beta_best, p=p_depth)
    x_raw = (qaoa_result.probabilities.argmax() >> np.arange(len(active_vars))) & 1
    x_raw = x_raw.astype(float)

    print(f"[Algorithm 2] raw bitstring feasible? "
          f"{verify_feasible(x_raw, conflicts)}  "
          f"raw served utility = {served_utility(qubo, x_raw):.3f}")

    # --- Algorithm 3: Constraint Repair ---
    x_feasible = repair(qubo, x_raw, requests, conflicts)
    print(f"[Algorithm 3] feasible after repair? {verify_feasible(x_feasible, conflicts)}")
    print(f"[Result] U(x_feasible) = {served_utility(qubo, x_feasible):.3f}  "
          f"(Eq. 1 objective value — this is the paper's reported metric)")

    # --- Exact baseline for sanity check on this tiny instance ---
    n = len(active_vars)
    best_brute = None
    best_val = -np.inf
    for mask in range(1 << n):
        x = np.array([(mask >> i) & 1 for i in range(n)], dtype=float)
        if not verify_feasible(x, conflicts):
            continue
        val = served_utility(qubo, x)
        if val > best_val:
            best_val, best_brute = val, x
    print(f"[Brute-force optimum] U(x*) = {best_val:.3f}  "
          f"(exact baseline for this {n}-qubit instance, cf. Gurobi in Table 6)")
    if best_val > 0:
        gap = 100 * (best_val - served_utility(qubo, x_feasible)) / best_val
        print(f"[Optimality gap] {gap:.2f}%  (cf. Table 7's 2.3-2.4%)")


if __name__ == "__main__":
    main()
