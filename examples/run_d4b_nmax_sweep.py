"""
D4-B: dose-response on DVQE partition width n_max.
  n_max in {10, 14, 18}, light GWO (10x25), p=1, n=25, default 8 trials.

  python3 -u examples/run_d4b_nmax_sweep.py --n-nodes 25 --n-trials 8
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
from qos_sim.qubo_builder import build_qubo, conflict_pairs_from_paths, served_utility
from qos_sim.baseline import exact_solver_highs
from qos_sim.repair import repair, verify_feasible
from qos_sim.qaoa_numpy import run_qaoa
from qos_sim.gwo_optimizer import optimize as gwo_optimize
from qos_sim.dvqe_partition import partition_by_weakest_edge

F_MIN = 0.95
P_DEPTH = 1
N_WOLVES = 10
MAX_ITER = 25
NMAX_LIST = (10, 14, 18)


def qaoa_raw(qubo, n_max, rng):
    n = len(qubo.variables)
    Q = qubo.Q
    if n > n_max:
        groups = partition_by_weakest_edge(Q, n_max)
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
    return x, len(groups)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-nodes", type=int, default=25)
    ap.add_argument("--n-trials", type=int, default=8)
    ap.add_argument("--seed0", type=int, default=42)
    ap.add_argument("--time-limit", type=float, default=5.0)
    args = ap.parse_args()
    out = Path(__file__).resolve().parents[1] / f"d4b_nmax_sweep_{args.n_nodes}.csv"
    fieldnames, rows = None, []
    print(
        f"=== D4-B n_max sweep  n={args.n_nodes} trials={args.n_trials} "
        f"n_max={list(NMAX_LIST)} GWO={N_WOLVES}x{MAX_ITER} p={P_DEPTH} ==="
    )
    for i in range(args.n_trials):
        seed = args.seed0 + i
        ns.set_qstate_formalism(QFormalism.DM)
        node_ids, links, graph = generate_topology(args.n_nodes, seed=seed)
        requests, paths, time_slots, path_resources, valid_paths = generate_requests_and_paths(
            graph, n_requests=args.n_nodes, k_paths=2, n_timeslots=3, seed=seed
        )
        network = build_topology(node_ids, links)
        trace = run_trace(network, paths, time_slots, link_params=links)
        tel = NetSquidTelemetry(trace)
        active, st = prune(
            requests, paths, time_slots, tel,
            fidelity_threshold=F_MIN, valid_paths=valid_paths, collision_pruning=False,
        )
        conflicts = conflict_pairs_from_paths(active, path_resources)
        qubo = build_qubo(active, requests, conflicts, lambda_margin=1.0)
        exact = exact_solver_highs(qubo, requests, conflicts, time_limit_s=args.time_limit)
        u_ilp = float(exact.served_utility)
        line = f"  seed={seed:4d} |A|={len(active)}"
        for n_max in NMAX_LIST:
            rng = np.random.default_rng(seed * 100 + n_max)
            t0 = time.perf_counter()
            x_raw, M = qaoa_raw(qubo, n_max, rng)
            t_q = time.perf_counter() - t0
            raw_feas = bool(verify_feasible(x_raw, conflicts))
            x_rep = repair(qubo, x_raw, requests, conflicts)
            u_rep = float(served_utility(qubo, x_rep))
            gap = (u_ilp - u_rep) / u_ilp * 100.0 if u_ilp > 1e-9 else 0.0
            rec = {
                "n_nodes": args.n_nodes,
                "trial_seed": seed,
                "size_A": len(active),
                "n_max": n_max,
                "M": M,
                "raw_feas": int(raw_feas),
                "U_repair": u_rep,
                "U_ilp": u_ilp,
                "gap_pct": gap,
                "t_qaoa_s": t_q,
            }
            rows.append(rec)
            if fieldnames is None:
                fieldnames = list(rec.keys())
                with out.open("w", newline="") as f:
                    w = csv.DictWriter(f, fieldnames=fieldnames)
                    w.writeheader()
                    w.writerow(rec)
            else:
                with out.open("a", newline="") as f:
                    csv.DictWriter(f, fieldnames=fieldnames).writerow(rec)
            line += (
                f"  nmax={n_max}:M={M}/raw={int(raw_feas)}/"
                f"gap={gap:.2f}%/t={t_q:.1f}s"
            )
        print(line)
    print("--- Summary by n_max ---")
    for n_max in NMAX_LIST:
        sub = [r for r in rows if r["n_max"] == n_max]
        n = len(sub)
        print(
            f"  n_max={n_max}: raw_feas={sum(r['raw_feas'] for r in sub)}/{n}  "
            f"mean_M={sum(r['M'] for r in sub)/n:.1f}  "
            f"mean_gap={sum(r['gap_pct'] for r in sub)/n:.2f}%  "
            f"mean_t={sum(r['t_qaoa_s'] for r in sub)/n:.1f}s"
        )
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
