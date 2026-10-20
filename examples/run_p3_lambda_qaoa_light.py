"""
P3 light: lambda margin x QAOA (reduced GWO budget).
  margins 1.0 and 1.1, n=25, default 15 trials
  GWO 10 wolves x 25 iter, p=1, max 10 qubits/partition

  python3 -u examples/run_p3_lambda_qaoa_light.py --n-nodes 25 --n-trials 15
"""
from __future__ import annotations
import argparse, csv, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import netsquid as ns
from netsquid.qubits import QFormalism

from qos_sim.topology_gen import generate_topology, generate_requests_and_paths
from qos_sim.pruning import prune
from qos_sim.simulator import NetSquidTelemetry, build_topology, run_trace
from qos_sim.qubo_builder import build_qubo, conflict_pairs_from_paths
from qos_sim.baseline import exact_solver_highs, verify_feasible, served_utility, repair
from qos_sim.qaoa_numpy import run_qaoa
from qos_sim.gwo_optimizer import optimize as gwo_optimize
from qos_sim.dvqe_partition import partition_by_weakest_edge

F_MIN = 0.95
P_DEPTH = 1
MAX_QUBITS = 10
N_WOLVES = 10
MAX_ITER = 25
MARGINS = (1.0, 1.1)


def qaoa_raw(qubo, conflicts, rng):
    n = len(qubo.variables)
    Q = qubo.Q
    if n > MAX_QUBITS:
        groups = partition_by_weakest_edge(Q, MAX_QUBITS)
    else:
        groups = [list(range(n))]
    x = np.zeros(n)
    for group in groups:
        sub_Q = Q[np.ix_(group, group)]
        def fitness(params, sub_Q=sub_Q):
            g = float(np.mean(params[:P_DEPTH]))
            b = float(np.mean(params[P_DEPTH:]))
            return run_qaoa(sub_Q, g, b, p=P_DEPTH).expectation
        gwo_r = gwo_optimize(
            fitness, dim=2 * P_DEPTH, n_wolves=N_WOLVES, max_iter=MAX_ITER, rng=rng
        )
        g_b = float(gwo_r.best_position[:P_DEPTH].mean())
        b_b = float(gwo_r.best_position[P_DEPTH:].mean())
        qr = run_qaoa(sub_Q, g_b, b_b, p=P_DEPTH)
        best = int(np.argmax(qr.probabilities))
        for li, gi in enumerate(group):
            x[gi] = (best >> li) & 1
    return x


def one_seed(n_nodes, seed, time_limit_s):
    ns.set_qstate_formalism(QFormalism.DM)
    node_ids, links, graph = generate_topology(n_nodes, seed=seed)
    requests, paths, time_slots, path_resources, valid_paths = generate_requests_and_paths(
        graph, n_requests=n_nodes, k_paths=2, n_timeslots=3, seed=seed
    )
    network = build_topology(node_ids, links)
    trace = run_trace(network, paths, time_slots, link_params=links)
    tel = NetSquidTelemetry(trace)
    active, stats = prune(
        requests, paths, time_slots, tel,
        fidelity_threshold=F_MIN, valid_paths=valid_paths, collision_pruning=False,
    )
    conflicts = conflict_pairs_from_paths(active, path_resources)
    rows = []
    for m in MARGINS:
        qubo = build_qubo(active, requests, conflicts, lambda_margin=m)
        exact = exact_solver_highs(qubo, requests, conflicts, time_limit_s=time_limit_s)
        t0 = time.perf_counter()
        rng = np.random.default_rng(seed + int(m * 100))
        x_raw = qaoa_raw(qubo, conflicts, rng)
        t_q = time.perf_counter() - t0
        raw_feas = verify_feasible(x_raw, conflicts)
        x_rep = repair(qubo, x_raw, requests, conflicts)
        u_rep = served_utility(qubo, x_rep)
        u_ilp = exact.served_utility
        gap = (u_ilp - u_rep) / u_ilp * 100.0 if u_ilp > 1e-9 else 0.0
        rows.append({
            "n_nodes": n_nodes, "trial_seed": seed, "lambda_margin": m,
            "size_A": len(active), "lambda_star": qubo.lambda_star,
            "raw_qaoa_feasible": int(raw_feas),
            "U_repair": u_rep, "U_ilp": u_ilp, "gap_pct": gap,
            "t_qaoa_s": t_q, "ilp_optimal": bool(exact.optimal),
        })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-nodes", type=int, default=25)
    ap.add_argument("--n-trials", type=int, default=15)
    ap.add_argument("--seed0", type=int, default=42)
    ap.add_argument("--time-limit", type=float, default=30.0)
    args = ap.parse_args()
    out = Path(__file__).resolve().parents[1] / f"p3_lambda_qaoa_light_{args.n_nodes}.csv"
    fieldnames, all_rows = None, []
    print(
        f"=== P3 light lambda-QAOA  n={args.n_nodes} trials={args.n_trials} "
        f"margins={list(MARGINS)} GWO={N_WOLVES}x{MAX_ITER} p={P_DEPTH} ==="
    )
    for i in range(args.n_trials):
        seed = args.seed0 + i
        rows = one_seed(args.n_nodes, seed, args.time_limit)
        for r in rows:
            all_rows.append(r)
            if fieldnames is None:
                fieldnames = list(r.keys())
                with out.open("w", newline="") as f:
                    w = csv.DictWriter(f, fieldnames=fieldnames)
                    w.writeheader()
                    w.writerow(r)
            else:
                with out.open("a", newline="") as f:
                    csv.DictWriter(f, fieldnames=fieldnames).writerow(r)
        by = {r["lambda_margin"]: r for r in rows}
        print(
            f"  seed={seed:4d}  "
            + "  ".join(
                f"m={m}:raw_feas={by[m]['raw_qaoa_feasible']} "
                f"gap={by[m]['gap_pct']:.2f}% t={by[m]['t_qaoa_s']:.1f}s"
                for m in MARGINS
            )
        )
    print("--- Summary by margin ---")
    for m in MARGINS:
        sub = [r for r in all_rows if r["lambda_margin"] == m]
        n = len(sub)
        feas = sum(r["raw_qaoa_feasible"] for r in sub)
        mean_gap = sum(r["gap_pct"] for r in sub) / n
        mean_t = sum(r["t_qaoa_s"] for r in sub) / n
        print(f"  m={m}: raw_feas={feas}/{n}  mean_gap={mean_gap:.2f}%  mean_t={mean_t:.1f}s")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
