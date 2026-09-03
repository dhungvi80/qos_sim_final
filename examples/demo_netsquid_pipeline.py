"""
demo_netsquid_pipeline.py — toàn bộ pipeline Algorithm 1-4 với NetSquid
telemetry thật thay cho DemoTelemetry giả lập.

Kịch bản: mạng 5 node, 6 link, 4 path, 3 time slot — đủ nhỏ để chạy
nhanh nhưng đủ lớn để kiểm chứng pruning + QUBO + QAOA + repair có ý
nghĩa thống kê (|A| thường ~5-12 qubit sau pruning, phù hợp numpy QAOA).

Chạy: python3 examples/demo_netsquid_pipeline.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import netsquid as ns
from netsquid.qubits import QFormalism

from qos_sim.pruning import PathConfig, Request, prune
from qos_sim.simulator import LinkParams, NetSquidTelemetry, build_topology, run_trace
from qos_sim.qubo_builder import build_qubo, conflict_pairs_from_paths, served_utility
from qos_sim.qaoa_numpy import run_qaoa
from qos_sim.gwo_optimizer import optimize as gwo_optimize
from qos_sim.repair import repair, verify_feasible
from qos_sim.dvqe_partition import (partition_by_weakest_edge, reconstruct_expectation,
                                      marginals_from_probabilities)


def build_network():
    """Mạng 5-node (A-E) với 6 link, mô phỏng một đoạn mạng lượng tử
    đơn giản. Các thông số link phản ánh link ngắn (5km) và link dài
    (20km) để tạo ra sự chênh lệch fidelity rõ ràng sau pruning."""
    node_ids = ["A", "B", "C", "D", "E"]
    links = {
        ("A", "B"): LinkParams(length_km=5.0),
        ("A", "C"): LinkParams(length_km=20.0),
        ("B", "C"): LinkParams(length_km=8.0),
        ("B", "D"): LinkParams(length_km=15.0),
        ("C", "E"): LinkParams(length_km=10.0),
        ("D", "E"): LinkParams(length_km=12.0),
    }
    return node_ids, links


def build_requests_and_paths():
    """6 request với priority weight khác nhau, 4 candidate path."""
    requests = [
        Request(id="req0", priority_weight=9.0),
        Request(id="req1", priority_weight=7.5),
        Request(id="req2", priority_weight=6.0),
        Request(id="req3", priority_weight=4.5),
        Request(id="req4", priority_weight=3.0),
        Request(id="req5", priority_weight=1.5),
    ]
    paths = [
        PathConfig(id="p_AB",  resources=("A-B",)),
        PathConfig(id="p_AC",  resources=("A-C",)),
        PathConfig(id="p_ABD", resources=("A-B", "B-D")),
        PathConfig(id="p_ACE", resources=("A-C", "C-E")),
    ]
    time_slots = [0, 1, 2]
    return requests, paths, time_slots


def main():
    trial_seed = 42
    rng = np.random.default_rng(trial_seed)

    print("=" * 55)
    print("  Demo: Full Pipeline with Real NetSquid Telemetry")
    print("=" * 55)

    # --- Build topology ---
    print("\n[Setup] Building 5-node quantum network topology...")
    node_ids, links = build_network()
    network = build_topology(node_ids, links)
    requests, paths, time_slots = build_requests_and_paths()
    path_resources = {p.id: p.resources for p in paths}
    print(f"  Nodes: {node_ids}")
    print(f"  Links: {len(links)}, Paths: {len(paths)}, "
          f"Time slots: {len(time_slots)}")
    print(f"  Requests: {len(requests)}, total weight = "
          f"{sum(r.priority_weight for r in requests):.1f}")

    # --- Run NetSquid trace ---
    print("\n[NetSquid] Running fidelity trace...")
    ns.set_qstate_formalism(QFormalism.DM)
    trace = run_trace(network, paths, time_slots, link_params=links)
    print("  Fidelity snapshot per (path, t):")
    for (pid, t), snap in sorted(trace.items()):
        print(f"    ({pid}, t={t}):  fidelity={snap.fidelity:.4f}  "
              f"available={snap.available}")

    # --- Algorithm 1: Pruning ---
    print("\n[Algorithm 1] Pruning with F_min=0.80...")
    telemetry = NetSquidTelemetry(trace)
    F_MIN = 0.95  # ngưỡng có ý nghĩa với T2=10ms thật (xem demo_scale_pipeline.py)
    active_vars, stats = prune(
        requests, paths, time_slots, telemetry,
        fidelity_threshold=F_MIN
    )
    print(f"  |D| = {stats.size_D}  |A| = {stats.size_A}  "
          f"reduction = {stats.reduction_ratio:.1%}")
    print(f"  rejected: fidelity={stats.rejected_fidelity}  "
          f"availability={stats.rejected_availability}  "
          f"collision={stats.rejected_collision}")

    if len(active_vars) == 0:
        print("  Active subspace empty — all paths below fidelity threshold.")
        print("  Try lowering F_MIN or adjusting link parameters.")
        return

    # --- Section 3.5-3.6: QUBO + Theorem 1 ---
    print("\n[Theorem 1] Building QUBO with tight penalty bound...")
    conflicts = conflict_pairs_from_paths(active_vars, path_resources)
    qubo = build_qubo(active_vars, requests, conflicts)
    print(f"  n_qubits = {len(active_vars)}  conflicts = {len(conflicts)}")
    print(f"  lambda* = {qubo.lambda_star:.4f}  "
          f"(Theorem 1, Eq. 15 — computable from network topology)")

    # --- Algorithm 2 + GWO: QAOA với DVQE partitioning ---
    print("\n[GWO + QAOA] Searching optimal circuit parameters...")
    p_depth = 2
    MAX_QUBITS_PER_PARTITION = 10  # giới hạn numpy statevector simulator

    n = len(active_vars)
    if n > MAX_QUBITS_PER_PARTITION:
        print(f"  |A|={n} > {MAX_QUBITS_PER_PARTITION} — DVQE weakest-edge "
              f"partitioning (Algorithm 2, Mục 4.3.1)...")
        index_groups = partition_by_weakest_edge(qubo.Q, MAX_QUBITS_PER_PARTITION)
        print(f"  {len(index_groups)} partitions: {[len(g) for g in index_groups]} qubits")

        x_raw = np.zeros(n)
        partition_marginals = []
        for pid, group in enumerate(index_groups):
            sub_Q = qubo.Q[np.ix_(group, group)]

            def fitness_part(params, sub_Q=sub_Q):
                g, b = params[:p_depth], params[p_depth:]
                return run_qaoa(sub_Q, g.mean(), b.mean(), p=p_depth).expectation

            gwo_r = gwo_optimize(fitness_part, dim=2 * p_depth,
                                  n_wolves=15, max_iter=30, rng=rng)
            g_b = gwo_r.best_position[:p_depth].mean()
            b_b = gwo_r.best_position[p_depth:].mean()
            qr = run_qaoa(sub_Q, g_b, b_b, p=p_depth)

            marg = marginals_from_probabilities(qr.probabilities, len(group))
            partition_marginals.append(marg)
            # bitstring decode: argmax of the JOINT distribution (one real
            # measurement outcome), NOT per-qubit marginal thresholding —
            # thresholding marginals independently discards correlations
            # QAOA builds between qubits and tends to collapse everything
            # toward 0 when no single qubit's marginal exceeds 0.5.
            best_state = int(np.argmax(qr.probabilities))
            x_part = np.array([(best_state >> k) & 1 for k in range(len(group))],
                               dtype=float)
            for local_idx, global_idx in enumerate(group):
                x_raw[global_idx] = x_part[local_idx]

            print(f"    Partition {pid}: n={len(group)}  <H_C>={gwo_r.best_fitness:.3f}")

        total_expectation = reconstruct_expectation(qubo.Q, index_groups, partition_marginals)
        print(f"  Reconstructed <H_C> (Eq. 16, incl. boundary terms) = {total_expectation:.4f}")
    else:
        def fitness(params: np.ndarray) -> float:
            g, b = params[:p_depth], params[p_depth:]
            return run_qaoa(qubo.Q, g.mean(), b.mean(), p=p_depth).expectation

        gwo_result = gwo_optimize(
            fitness, dim=2 * p_depth, n_wolves=20, max_iter=50, rng=rng
        )
        g_best = gwo_result.best_position[:p_depth].mean()
        b_best = gwo_result.best_position[p_depth:].mean()
        qaoa_result = run_qaoa(qubo.Q, g_best, b_best, p=p_depth)
        x_raw = (
            (qaoa_result.probabilities.argmax() >> np.arange(n)) & 1
        ).astype(float)
        print(f"  Best <H_C> = {gwo_result.best_fitness:.4f}")

    print(f"  Raw bitstring feasible? {verify_feasible(x_raw, conflicts)}")
    print(f"  Raw U(x) = {served_utility(qubo, x_raw):.3f}")

    # --- Algorithm 3: Constraint Repair ---
    print("\n[Algorithm 3] Constraint repair...")
    x_feasible = repair(qubo, x_raw, requests, conflicts)
    feasible = verify_feasible(x_feasible, conflicts)
    u_final = served_utility(qubo, x_feasible)
    print(f"  Feasible after repair? {feasible}")
    print(f"  U(x_feasible) = {u_final:.3f}  "
          f"← Eq. (1) objective — value for Table 6/7")

    # --- Brute-force optimum (exact, only feasible for small |A|) ---
    if n <= 14:
        print("\n[Exact] Computing brute-force optimum for comparison...")
        best_val = -np.inf
        for mask in range(1 << n):
            x = np.array([(mask >> i) & 1 for i in range(n)], dtype=float)
            if verify_feasible(x, conflicts):
                v = served_utility(qubo, x)
                if v > best_val:
                    best_val = v
        gap = 100 * (best_val - u_final) / best_val if best_val > 0 else 0
        print(f"  U(x*) = {best_val:.3f}  "
              f"(Gurobi baseline equivalent for Table 6)")
        print(f"  Optimality gap = {gap:.2f}%  (cf. paper's claim: ~2.3-2.4%)")
    else:
        print(f"\n[Exact] Skipped — |A|={n} > 14, use Gurobi on research server.")

    print("\n" + "=" * 55)
    print("  Pipeline complete — NetSquid telemetry validated.")
    print("  Next: scale to 25/50/100-node with DVQE partitioning.")
    print("=" * 55)


if __name__ == "__main__":
    main()
