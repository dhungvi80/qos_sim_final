"""
demo_scale_pipeline.py — chạy toàn bộ pipeline Algorithm 1-4 ở quy mô
25/50/100-node, dùng topology_gen.py (thay cho scenario 5-node viết tay).

Chạy:
    python3 examples/demo_scale_pipeline.py --n-nodes 25
    python3 examples/demo_scale_pipeline.py --n-nodes 50
    python3 examples/demo_scale_pipeline.py --n-nodes 100

Log ra CSV để tổng hợp thành Table 6-8 (xem save_result()).
"""

import argparse
import csv
import sys
import time
import tracemalloc
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import netsquid as ns
from netsquid.qubits import QFormalism

from qos_sim.topology_gen import generate_topology, generate_requests_and_paths
from qos_sim.pruning import prune
from qos_sim.simulator import NetSquidTelemetry, build_topology, run_trace
from qos_sim.qubo_builder import build_qubo, conflict_pairs_from_paths, served_utility
from qos_sim.qaoa_numpy import run_qaoa
from qos_sim.gwo_optimizer import optimize as gwo_optimize
from qos_sim.repair import repair, verify_feasible
from qos_sim.dvqe_partition import (partition_by_weakest_edge, reconstruct_expectation,
                                      marginals_from_probabilities)
from qos_sim.baseline import (greedy_schedule, simulated_annealing,
                                tabu_search, genetic_algorithm, ant_colony_optimization,
                                random_init_and_repair)

MAX_QUBITS_PER_PARTITION = 10
F_MIN = 0.95  # với T2=10ms thật, fidelity dao động ~0.83-0.99 theo hop count;
              # 0.80 (giá trị cũ, hiệu chỉnh cho depolar_rate tự chọn trước đây)
              # gần như không bao giờ kích hoạt nữa — 0.95 mới có ý nghĩa lọc thật
P_DEPTH = 3  # tăng từ 2 lên 3 — QAOA thua Greedy/SA ở p=2, thử tăng độ sâu


def run_scenario(n_nodes: int, trial_seed: int = 42, verbose: bool = True) -> dict:
    tracemalloc.start()
    t_start = time.perf_counter()
    rng = np.random.default_rng(trial_seed)

    node_ids, links, graph = generate_topology(n_nodes, seed=trial_seed)
    requests, paths, time_slots, path_resources, valid_paths = generate_requests_and_paths(
        graph, n_requests=n_nodes, k_paths=2, n_timeslots=3, seed=trial_seed
    )
    if verbose:
        print(f"[Setup] n_nodes={n_nodes}  edges={graph.number_of_edges()}  "
              f"requests={len(requests)}  candidate_paths={len(paths)}")

    ns.set_qstate_formalism(QFormalism.DM)
    network = build_topology(node_ids, links)
    trace = run_trace(network, paths, time_slots, link_params=links)
    telemetry = NetSquidTelemetry(trace)

    active_vars, stats = prune(
        requests, paths, time_slots, telemetry,
        fidelity_threshold=F_MIN, valid_paths=valid_paths,
    )
    if verbose:
        print(f"[Algorithm 1] |D|={stats.size_D}  |A|={stats.size_A}  "
              f"reduction={stats.reduction_ratio:.1%}  "
              f"rejected(fid/avail/coll)={stats.rejected_fidelity}/"
              f"{stats.rejected_availability}/{stats.rejected_collision}")

    if len(active_vars) == 0:
        return {"n_nodes": n_nodes, "error": "empty active subspace"}

    conflicts = conflict_pairs_from_paths(active_vars, path_resources)
    qubo = build_qubo(active_vars, requests, conflicts)
    if verbose:
        print(f"[Theorem 1] n_qubits={len(active_vars)}  conflicts={len(conflicts)}  "
              f"lambda*={qubo.lambda_star:.4f}")

    n = len(active_vars)
    if n > MAX_QUBITS_PER_PARTITION:
        index_groups = partition_by_weakest_edge(qubo.Q, MAX_QUBITS_PER_PARTITION)
    else:
        index_groups = [list(range(n))]
    if verbose:
        print(f"[Algorithm 2] {len(index_groups)} partitions: "
              f"{[len(g) for g in index_groups]} qubits")

    x_raw = np.zeros(n)
    partition_marginals = []
    for group in index_groups:
        sub_Q = qubo.Q[np.ix_(group, group)]

        def fitness_part(params, sub_Q=sub_Q):
            g, b = params[:P_DEPTH], params[P_DEPTH:]
            return run_qaoa(sub_Q, g.mean(), b.mean(), p=P_DEPTH).expectation

        gwo_r = gwo_optimize(fitness_part, dim=2 * P_DEPTH,
                              n_wolves=30, max_iter=80, rng=rng)
        g_b, b_b = gwo_r.best_position[:P_DEPTH].mean(), gwo_r.best_position[P_DEPTH:].mean()
        qr = run_qaoa(sub_Q, g_b, b_b, p=P_DEPTH)

        marg = marginals_from_probabilities(qr.probabilities, len(group))
        partition_marginals.append(marg)
        best_state = int(np.argmax(qr.probabilities))
        x_part = np.array([(best_state >> k) & 1 for k in range(len(group))], dtype=float)
        for local_idx, global_idx in enumerate(group):
            x_raw[global_idx] = x_part[local_idx]

    total_expectation = reconstruct_expectation(qubo.Q, index_groups, partition_marginals)

    # --- Table 7 (ablation): raw QAOA (trước repair) ---
    raw_feasible = verify_feasible(x_raw, conflicts)
    raw_utility = served_utility(qubo, x_raw)

    x_feasible = repair(qubo, x_raw, requests, conflicts)
    feasible = verify_feasible(x_feasible, conflicts)
    u_final = served_utility(qubo, x_feasible)
    total_weight = sum(r.priority_weight for r in requests)

    # --- Ablation: Random init + Repair (isolates classical repair value) ---
    random_result = random_init_and_repair(
        qubo, requests, conflicts, seed=trial_seed, p_activate=0.5
    )

    # --- Table 6 (baseline so sánh): Greedy + SA + Tabu + GA + ACO ---
    greedy_result = greedy_schedule(qubo, requests, conflicts)
    sa_result = simulated_annealing(qubo, conflicts, n_iters=2000, seed=trial_seed)
    tabu_result = tabu_search(qubo, conflicts, n_iters=500, seed=trial_seed)
    ga_result = genetic_algorithm(qubo, conflicts, population_size=40,
                                   n_generations=100, seed=trial_seed)
    aco_result = ant_colony_optimization(qubo, conflicts, n_ants=30,
                                          n_iterations=60, seed=trial_seed)

    peak_mem_mb = tracemalloc.get_traced_memory()[1] / (1024 * 1024)
    tracemalloc.stop()

    runtime_s = time.perf_counter() - t_start
    if verbose:
        print(f"[Algorithm 3] feasible={feasible}  U(x)={u_final:.3f}  "
              f"(max possible={total_weight:.1f})")
        print(f"[Ablation] Random+Repair={random_result.served_utility:.3f}  "
              f"QAOA+Repair={u_final:.3f}  Greedy={greedy_result.served_utility:.3f}")
        print(f"[Baseline] SA={sa_result.served_utility:.3f}  "
              f"Tabu={tabu_result.served_utility:.3f}  "
              f"GA={ga_result.served_utility:.3f}  ACO={aco_result.served_utility:.3f}")
        print(f"[Runtime] {runtime_s:.2f}s   [Peak memory] {peak_mem_mb:.1f} MB")

    return {
        "n_nodes": n_nodes,
        "trial_seed": trial_seed,
        "size_D": stats.size_D,
        "size_A": stats.size_A,
        "reduction_ratio": stats.reduction_ratio,
        "rejected_fidelity": stats.rejected_fidelity,
        "rejected_availability": stats.rejected_availability,
        "rejected_collision": stats.rejected_collision,
        "n_conflicts": len(conflicts),
        "lambda_star": qubo.lambda_star,
        "n_partitions": len(index_groups),
        "reconstructed_HC": total_expectation,
        "feasible": feasible,
        "U_x": u_final,
        # Table 7 — ablation (raw QAOA trước repair, cùng 1 lần chạy)
        "raw_qaoa_feasible": raw_feasible,
        "raw_qaoa_U_x": raw_utility,
        # Ablation: Random init + Repair
        "random_repair_feasible": random_result.feasible,
        "random_repair_U_x": random_result.served_utility,
        "random_repair_runtime_s": random_result.runtime_s,
        # Table 6 — baseline (cùng qubo/conflicts, so sánh công bằng)
        "greedy_feasible": greedy_result.feasible,
        "greedy_U_x": greedy_result.served_utility,
        "greedy_runtime_s": greedy_result.runtime_s,
        "sa_feasible": sa_result.feasible,
        "sa_U_x": sa_result.served_utility,
        "sa_runtime_s": sa_result.runtime_s,
        "tabu_feasible": tabu_result.feasible,
        "tabu_U_x": tabu_result.served_utility,
        "tabu_runtime_s": tabu_result.runtime_s,
        "ga_feasible": ga_result.feasible,
        "ga_U_x": ga_result.served_utility,
        "ga_runtime_s": ga_result.runtime_s,
        "aco_feasible": aco_result.feasible,
        "aco_U_x": aco_result.served_utility,
        "aco_runtime_s": aco_result.runtime_s,
        "total_weight": total_weight,
        "runtime_s": runtime_s,
        "peak_memory_mb": peak_mem_mb,
    }



def save_result(result: dict, csv_path: str = "results.csv"):
    file_exists = Path(csv_path).exists()
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(result.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-nodes", type=int, default=25, choices=[25, 50, 100])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--csv", type=str, default="results.csv")
    args = parser.parse_args()

    result = run_scenario(args.n_nodes, trial_seed=args.seed)
    save_result(result, args.csv)
    print(f"\nSaved to {args.csv}: {result}")
