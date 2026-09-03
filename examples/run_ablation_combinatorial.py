"""
run_ablation_combinatorial.py — Ablation Random+Repair vs QAOA+Repair vs Greedy
không cần NetSquid (dùng SyntheticTelemetry + topology_gen).

Mục tiêu reviewer: định lượng đóng góp của QAOA so với classical repair.

Chạy:
    python3 examples/run_ablation_combinatorial.py --n-nodes 25 --n-trials 30
    python3 examples/run_ablation_combinatorial.py --n-nodes 50 --n-trials 30
"""
from __future__ import annotations

import argparse
import csv
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from qos_sim.topology_gen import generate_topology, generate_requests_and_paths
from qos_sim.pruning import prune, NetworkTelemetry, PathConfig
from qos_sim.qubo_builder import build_qubo, conflict_pairs_from_paths, served_utility
from qos_sim.qaoa_numpy import run_qaoa
from qos_sim.gwo_optimizer import optimize as gwo_optimize
from qos_sim.repair import repair, verify_feasible
from qos_sim.dvqe_partition import (
    partition_by_weakest_edge, reconstruct_expectation, marginals_from_probabilities,
)
from qos_sim.baseline import greedy_schedule, random_init_and_repair

MAX_QUBITS_PER_PARTITION = 10
F_MIN = 0.95
P_DEPTH = 3


class SyntheticTelemetry(NetworkTelemetry):
    def __init__(self, trial_seed: int, fidelity_floor: float = 0.5,
                 unavailable_prob: float = 0.1):
        self._rng_seed = trial_seed
        self._fidelity_floor = fidelity_floor
        self._unavailable_prob = unavailable_prob

    def _rng_for(self, path: PathConfig, t: int):
        import random
        return random.Random(f"{self._rng_seed}:{path.id}:{t}")

    def fidelity(self, path: PathConfig, t: int) -> float:
        rng = self._rng_for(path, t)
        return self._fidelity_floor + rng.random() * (1 - self._fidelity_floor)

    def is_available(self, path: PathConfig, t: int) -> bool:
        rng = self._rng_for(path, t)
        rng.random()
        return rng.random() > self._unavailable_prob

    def has_collision(self, path, t, reserved):
        for resource in path.resources:
            holders = reserved.get((resource, t), set())
            if holders and path.id not in holders:
                return True
        return False


def run_one(n_nodes: int, trial_seed: int) -> dict:
    rng = np.random.default_rng(trial_seed)
    node_ids, links, graph = generate_topology(n_nodes, seed=trial_seed)
    requests, paths, time_slots, path_resources, valid_paths = generate_requests_and_paths(
        graph, n_requests=n_nodes, k_paths=2, n_timeslots=3, seed=trial_seed
    )
    telemetry = SyntheticTelemetry(trial_seed)
    active_vars, stats = prune(
        requests, paths, time_slots, telemetry,
        fidelity_threshold=F_MIN, valid_paths=valid_paths,
    )
    if len(active_vars) == 0:
        return {"n_nodes": n_nodes, "trial_seed": trial_seed, "error": "empty A"}

    conflicts = conflict_pairs_from_paths(active_vars, path_resources)
    qubo = build_qubo(active_vars, requests, conflicts)
    n = len(active_vars)

    # QAOA (numpy) + partition
    if n > MAX_QUBITS_PER_PARTITION:
        index_groups = partition_by_weakest_edge(qubo.Q, MAX_QUBITS_PER_PARTITION)
    else:
        index_groups = [list(range(n))]

    x_raw = np.zeros(n)
    for group in index_groups:
        sub_Q = qubo.Q[np.ix_(group, group)]

        def fitness_part(params, sub_Q=sub_Q):
            g, b = params[:P_DEPTH], params[P_DEPTH:]
            return run_qaoa(sub_Q, g.mean(), b.mean(), p=P_DEPTH).expectation

        gwo_r = gwo_optimize(fitness_part, dim=2 * P_DEPTH,
                             n_wolves=15, max_iter=40, rng=rng)
        g_b = gwo_r.best_position[:P_DEPTH].mean()
        b_b = gwo_r.best_position[P_DEPTH:].mean()
        qr = run_qaoa(sub_Q, g_b, b_b, p=P_DEPTH)
        best_state = int(np.argmax(qr.probabilities))
        x_part = np.array([(best_state >> k) & 1 for k in range(len(group))], dtype=float)
        for local_idx, global_idx in enumerate(group):
            x_raw[global_idx] = x_part[local_idx]

    raw_feasible = verify_feasible(x_raw, conflicts)
    raw_u = served_utility(qubo, x_raw)

    x_qaoa = repair(qubo, x_raw, requests, conflicts)
    qaoa_u = served_utility(qubo, x_qaoa)
    qaoa_feas = verify_feasible(x_qaoa, conflicts)

    rand_r = random_init_and_repair(qubo, requests, conflicts, seed=trial_seed)
    greedy_r = greedy_schedule(qubo, requests, conflicts)

    return {
        "n_nodes": n_nodes,
        "trial_seed": trial_seed,
        "size_A": stats.size_A,
        "size_D": stats.size_D,
        "reduction_ratio": stats.reduction_ratio,
        "n_conflicts": len(conflicts),
        "raw_qaoa_U_x": raw_u,
        "raw_qaoa_feasible": raw_feasible,
        "qaoa_repair_U_x": qaoa_u,
        "qaoa_repair_feasible": qaoa_feas,
        "random_repair_U_x": rand_r.served_utility,
        "random_repair_feasible": rand_r.feasible,
        "greedy_U_x": greedy_r.served_utility,
        "greedy_feasible": greedy_r.feasible,
        "qaoa_eq_greedy": abs(qaoa_u - greedy_r.served_utility) < 1e-9,
        "random_eq_greedy": abs(rand_r.served_utility - greedy_r.served_utility) < 1e-9,
        "qaoa_eq_random": abs(qaoa_u - rand_r.served_utility) < 1e-9,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-nodes", type=int, default=25, choices=[25, 50, 100])
    parser.add_argument("--n-trials", type=int, default=30)
    parser.add_argument("--base-seed", type=int, default=42)
    args = parser.parse_args()

    rows = []
    t0 = time.perf_counter()
    for i in range(args.n_trials):
        seed = args.base_seed + i
        r = run_one(args.n_nodes, seed)
        if "error" in r:
            print(f"  trial {i} seed={seed}: SKIP {r['error']}")
            continue
        rows.append(r)
        print(f"  [{i+1}/{args.n_trials}] seed={seed}  "
              f"|A|={r['size_A']}  "
              f"Rand={r['random_repair_U_x']:.1f}  "
              f"QAOA={r['qaoa_repair_U_x']:.1f}  "
              f"Greedy={r['greedy_U_x']:.1f}  "
              f"Q=G:{r['qaoa_eq_greedy']} R=G:{r['random_eq_greedy']}",
              flush=True)

    if not rows:
        print("No successful trials.")
        return

    out = f"ablation_raw_{args.n_nodes}.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    def mean_sd(key):
        vals = [float(r[key]) for r in rows]
        return statistics.mean(vals), (statistics.stdev(vals) if len(vals) > 1 else 0.0)

    print(f"\n=== Ablation n={args.n_nodes}  N={len(rows)} trials ===")
    for label, key in [
        ("Random+Repair", "random_repair_U_x"),
        ("QAOA+Repair", "qaoa_repair_U_x"),
        ("Greedy", "greedy_U_x"),
        ("Raw QAOA", "raw_qaoa_U_x"),
    ]:
        m, s = mean_sd(key)
        print(f"  {label:16s}  {m:.2f} ± {s:.2f}")

    n_eq_qg = sum(1 for r in rows if r["qaoa_eq_greedy"])
    n_eq_rg = sum(1 for r in rows if r["random_eq_greedy"])
    n_eq_qr = sum(1 for r in rows if r["qaoa_eq_random"])
    print(f"  QAOA+Repair == Greedy : {n_eq_qg}/{len(rows)}")
    print(f"  Random+Repair == Greedy: {n_eq_rg}/{len(rows)}")
    print(f"  QAOA+Repair == Random  : {n_eq_qr}/{len(rows)}")
    print(f"  elapsed {time.perf_counter()-t0:.1f}s  → {out}")


if __name__ == "__main__":
    main()
