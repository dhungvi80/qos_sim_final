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
                                random_init_and_repair, exact_solver_highs)

MAX_QUBITS_PER_PARTITION = 10
F_MIN = 0.95  # với T2=10ms thật, fidelity dao động ~0.83-0.99 theo hop count;
              # 0.80 (giá trị cũ, hiệu chỉnh cho depolar_rate tự chọn trước đây)
              # gần như không bao giờ kích hoạt nữa — 0.95 mới có ý nghĩa lọc thật
EXACT_TIME_LIMIT_S = 60.0  # giới hạn thời gian ILP; quá hạn => exact_optimal=False, KHÔNG báo gap
COLLISION_PRUNING = False  # False = chỉ lọc đơn biến (fidelity/availability), bảo toàn nghiệm tối ưu
P_DEPTH = 3  # tăng từ 2 lên 3 — QAOA thua Greedy/SA ở p=2, thử tăng độ sâu


def run_scenario(n_nodes: int, trial_seed: int = 42, verbose: bool = True,
                 load: float = 1.0, skip_qaoa: bool = False,
                 exact_time_limit_s: float | None = None) -> dict:
    """load: n_requests = round(load * n_nodes). load=1.0 reproduces the original
    setting exactly (same RNG stream). skip_qaoa=True skips Algorithms 2-3 on the
    QAOA branch (GWO/QAOA/Repair) and leaves its columns NaN; every classical
    baseline and the ILP are unchanged because they use their own seeds."""
    if exact_time_limit_s is None:
        exact_time_limit_s = EXACT_TIME_LIMIT_S
    n_requests = max(1, int(round(load * n_nodes)))
    tracemalloc.start()
    t_start = time.perf_counter()
    rng = np.random.default_rng(trial_seed)

    node_ids, links, graph = generate_topology(n_nodes, seed=trial_seed)
    requests, paths, time_slots, path_resources, valid_paths = generate_requests_and_paths(
        graph, n_requests=n_requests, k_paths=2, n_timeslots=3, seed=trial_seed
    )
    if verbose:
        print(f"[Setup] n_nodes={n_nodes}  edges={graph.number_of_edges()}  "
              f"requests={len(requests)}  candidate_paths={len(paths)}  load={load:g}")

    ns.set_qstate_formalism(QFormalism.DM)
    network = build_topology(node_ids, links)
    trace = run_trace(network, paths, time_slots, link_params=links)
    telemetry = NetSquidTelemetry(trace)

    active_vars, stats = prune(
        requests, paths, time_slots, telemetry,
        fidelity_threshold=F_MIN, valid_paths=valid_paths,
        collision_pruning=COLLISION_PRUNING,
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

    nan = float("nan")
    total_expectation, raw_feasible, raw_utility = nan, None, nan
    feasible, u_final = None, nan
    if not skip_qaoa:
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

    # --- Baseline + Algorithm 3: mọi phương pháp metaheuristic đều qua Repair để
    # so sánh công bằng (các cột *_U_x / *_feasible ở trên vẫn là kết quả THÔ).
    # Đặt SAU runtime_s để runtime_s giữ nguyên ý nghĩa như các lần chạy trước.
    def _repaired(res):
        x_rep = repair(qubo, res.x, requests, conflicts)
        return verify_feasible(x_rep, conflicts), served_utility(qubo, x_rep)

    sa_rep_feas, sa_rep_u = _repaired(sa_result)
    tabu_rep_feas, tabu_rep_u = _repaired(tabu_result)
    ga_rep_feas, ga_rep_u = _repaired(ga_result)
    aco_rep_feas, aco_rep_u = _repaired(aco_result)

    # --- Nghiệm chính xác (ILP, HiGHS) trên đúng qubo/conflicts của trial này ---
    exact_result = exact_solver_highs(qubo, requests, conflicts,
                                      time_limit_s=exact_time_limit_s)

    def _gap(u):
        # optimality gap (%) so với ILP; NaN nếu ILP không chứng minh được tối ưu
        if exact_result.optimal and exact_result.served_utility > 1e-9:
            return 100.0 * (exact_result.served_utility - u) / exact_result.served_utility
        return float("nan")
    def _gap_ub(u):
        # UPPER bound on the optimality gap (%), valid even if the ILP hit its time
        # limit: (dual bound - U) / dual bound. Equals _gap(u) when ILP is optimal.
        b = exact_result.bound
        if np.isfinite(b) and b > 1e-9 and np.isfinite(u):
            return max(0.0, 100.0 * (b - u) / b)
        return float("nan")
    if verbose:
        print(f"[Algorithm 3] feasible={feasible}  U(x)={u_final:.3f}  "
              f"(max possible={total_weight:.1f})" + ("  [QAOA skipped]" if skip_qaoa else ""))
        print(f"[Ablation] Random+Repair={random_result.served_utility:.3f}  "
              f"QAOA+Repair={u_final:.3f}  Greedy={greedy_result.served_utility:.3f}")
        print(f"[Baseline] SA={sa_result.served_utility:.3f}  "
              f"Tabu={tabu_result.served_utility:.3f}  "
              f"GA={ga_result.served_utility:.3f}  ACO={aco_result.served_utility:.3f}")
        print(f"[Runtime] {runtime_s:.2f}s   [Peak memory] {peak_mem_mb:.1f} MB")

    return {
        "n_nodes": n_nodes,
        "n_requests": n_requests,
        "load": load,
        "skip_qaoa": skip_qaoa,
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
        # Baseline + Repair (Algorithm 3) — so sánh công bằng
        "sa_repair_feasible": sa_rep_feas,
        "sa_repair_U_x": sa_rep_u,
        "tabu_repair_feasible": tabu_rep_feas,
        "tabu_repair_U_x": tabu_rep_u,
        "ga_repair_feasible": ga_rep_feas,
        "ga_repair_U_x": ga_rep_u,
        "aco_repair_feasible": aco_rep_feas,
        "aco_repair_U_x": aco_rep_u,
        # Nghiệm chính xác (ILP/HiGHS) và optimality gap (%)
        "exact_optimal": exact_result.optimal,
        "exact_U_x": exact_result.served_utility,
        "exact_runtime_s": exact_result.runtime_s,
        "gap_qaoa_repair_pct": _gap(u_final),
        "gap_random_repair_pct": _gap(random_result.served_utility),
        "gap_greedy_pct": _gap(greedy_result.served_utility),
        "gap_sa_repair_pct": _gap(sa_rep_u),
        "gap_tabu_repair_pct": _gap(tabu_rep_u),
        "gap_ga_repair_pct": _gap(ga_rep_u),
        "gap_aco_repair_pct": _gap(aco_rep_u),
        # Phase 1: dual bound and valid gap upper bounds (finite even on ILP timeout)
        "exact_bound": exact_result.bound,
        "gap_ub_qaoa_repair_pct": _gap_ub(u_final),
        "gap_ub_random_repair_pct": _gap_ub(random_result.served_utility),
        "gap_ub_greedy_pct": _gap_ub(greedy_result.served_utility),
        "gap_ub_sa_repair_pct": _gap_ub(sa_rep_u),
        "gap_ub_tabu_repair_pct": _gap_ub(tabu_rep_u),
        "gap_ub_ga_repair_pct": _gap_ub(ga_rep_u),
        "gap_ub_aco_repair_pct": _gap_ub(aco_rep_u),
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
    parser.add_argument("--load", type=float, default=1.0,
                        help="n_requests = round(load * n_nodes); 1.0 = original setting")
    parser.add_argument("--skip-qaoa", action="store_true",
                        help="bỏ nhánh QAOA (GWO/QAOA/Repair); baseline và ILP giữ nguyên")
    parser.add_argument("--exact-time-limit", type=float, default=None,
                        help=f"giới hạn thời gian ILP (s), mặc định {EXACT_TIME_LIMIT_S:g}")
    args = parser.parse_args()

    result = run_scenario(args.n_nodes, trial_seed=args.seed, load=args.load,
                          skip_qaoa=args.skip_qaoa, exact_time_limit_s=args.exact_time_limit)
    save_result(result, args.csv)
    print(f"\nSaved to {args.csv}: {result}")
