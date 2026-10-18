"""
D5-C / E1-C: pruning ablation with correct Design Property 1 test.
ILP on D with variables outside A fixed to 0.
"""
from __future__ import annotations
import argparse, csv, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import netsquid as ns
from netsquid.qubits import QFormalism
from qos_sim.topology_gen import generate_topology, generate_requests_and_paths
from qos_sim.pruning import prune, Variable
from qos_sim.simulator import NetSquidTelemetry, build_topology, run_trace
from qos_sim.qubo_builder import build_qubo, conflict_pairs_from_paths, served_utility
from qos_sim.baseline import exact_solver_highs, greedy_schedule

F_MIN = 0.95
OUT_NAME = "d5c_pruning_ablation_{n}.csv"

def _all_variables(requests, paths, time_slots, valid_paths):
    paths_by_id = {p.id: p for p in paths}
    active = []
    for r in requests:
        if valid_paths is None:
            cand = list(paths)
        else:
            cand = [paths_by_id[pid] for pid in sorted(valid_paths.get(r.id, ()))
                    if pid in paths_by_id]
        for p in cand:
            for tslot in time_slots:
                active.append(Variable(r.id, p.id, tslot))
    return active

def _ilp_forced_zero(active_D, requests, path_resources, active_A, time_limit_s: float):
    from scipy.optimize import Bounds, LinearConstraint, milp
    from scipy.sparse import coo_matrix
    conflicts = conflict_pairs_from_paths(active_D, path_resources)
    qubo = build_qubo(active_D, requests, conflicts)
    n = len(active_D)
    if n == 0:
        return 0.0, 0.0, True
    A_keys = {v.key() for v in active_A}
    lb = np.zeros(n)
    ub = np.ones(n)
    for i, v in enumerate(active_D):
        if v.key() not in A_keys:
            ub[i] = 0.0
    weight_of = {r.id: r.priority_weight for r in requests}
    c = -np.array([weight_of[v.request_id] for v in active_D], dtype=float)
    constraints = []
    if conflicts:
        pairs = np.asarray(conflicts, dtype=int)
        rows = np.repeat(np.arange(len(pairs)), 2)
        Amat = coo_matrix(
            (np.ones(2 * len(pairs)), (rows, pairs.ravel())),
            shape=(len(pairs), n),
        ).tocsr()
        constraints = [LinearConstraint(Amat, -np.inf, 1.0)]
    t0 = time.perf_counter()
    res = milp(
        c=c, integrality=np.ones(n), bounds=Bounds(lb, ub),
        constraints=constraints, options={"time_limit": float(time_limit_s)},
    )
    runtime = time.perf_counter() - t0
    if res.x is None:
        return float("nan"), runtime, False
    x = np.round(res.x)
    u = served_utility(qubo, x)
    return float(u), runtime, bool(res.status == 0)

def one_trial(n_nodes: int, seed: int, time_limit_s: float) -> dict:
    ns.set_qstate_formalism(QFormalism.DM)
    node_ids, links, graph = generate_topology(n_nodes, seed=seed)
    requests, paths, time_slots, path_resources, valid_paths = generate_requests_and_paths(
        graph, n_requests=n_nodes, k_paths=2, n_timeslots=3, seed=seed
    )
    network = build_topology(node_ids, links)
    trace = run_trace(network, paths, time_slots, link_params=links)
    telemetry = NetSquidTelemetry(trace)
    active_A, stats = prune(
        requests, paths, time_slots, telemetry,
        fidelity_threshold=F_MIN, valid_paths=valid_paths, collision_pruning=False,
    )
    active_D = _all_variables(requests, paths, time_slots, valid_paths)
    conflicts_A = conflict_pairs_from_paths(active_A, path_resources)
    qubo_A = build_qubo(active_A, requests, conflicts_A)
    exact_A = exact_solver_highs(qubo_A, requests, conflicts_A, time_limit_s=time_limit_s)
    conflicts_D = conflict_pairs_from_paths(active_D, path_resources)
    qubo_D = build_qubo(active_D, requests, conflicts_D)
    exact_D_free = exact_solver_highs(qubo_D, requests, conflicts_D, time_limit_s=time_limit_s)
    u_fz, t_fz, opt_fz = _ilp_forced_zero(
        active_D, requests, path_resources, active_A, time_limit_s
    )
    greedy_A = greedy_schedule(qubo_A, requests, conflicts_A)
    u_match = (
        exact_A.optimal and opt_fz
        and abs(exact_A.served_utility - u_fz) < 1e-5
    )
    return {
        "n_nodes": n_nodes, "trial_seed": seed,
        "size_D": len(active_D), "size_A": len(active_A),
        "n_conflicts_D": len(conflicts_D), "n_conflicts_A": len(conflicts_A),
        "rho": 1.0 - (len(active_A) / len(active_D) if active_D else 0.0),
        "ilp_A_U": exact_A.served_utility,
        "ilp_A_time_s": exact_A.runtime_s,
        "ilp_A_optimal": bool(exact_A.optimal),
        "ilp_D_free_U": exact_D_free.served_utility,
        "ilp_D_free_time_s": exact_D_free.runtime_s,
        "ilp_D_free_optimal": bool(exact_D_free.optimal),
        "ilp_D_forced_U": u_fz,
        "ilp_D_forced_time_s": t_fz,
        "ilp_D_forced_optimal": bool(opt_fz),
        "greedy_A_U": greedy_A.served_utility,
        "U_match_DP1": u_match,
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-nodes", type=int, default=25)
    ap.add_argument("--n-trials", type=int, default=30)
    ap.add_argument("--seed0", type=int, default=42)
    ap.add_argument("--time-limit", type=float, default=30.0)
    args = ap.parse_args()
    out = Path(__file__).resolve().parents[1] / OUT_NAME.format(n=args.n_nodes)
    fieldnames, rows = None, []
    print(f"=== D5-C E1-C  n={args.n_nodes} trials={args.n_trials} ===")
    for i in range(args.n_trials):
        seed = args.seed0 + i
        r = one_trial(args.n_nodes, seed, args.time_limit)
        rows.append(r)
        if fieldnames is None:
            fieldnames = list(r.keys())
            with out.open("w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=fieldnames)
                w.writeheader(); w.writerow(r)
        else:
            with out.open("a", newline="") as f:
                csv.DictWriter(f, fieldnames=fieldnames).writerow(r)
        print(
            f"  seed={seed:4d} |D|={r['size_D']:4d} |A|={r['size_A']:4d} "
            f"rho={r['rho']:.3f}  "
            f"U_A={r['ilp_A_U']:.3f}  U_forced={r['ilp_D_forced_U']:.3f}  "
            f"U_free={r['ilp_D_free_U']:.3f}  "
            f"t_A={r['ilp_A_time_s']:.3f}s t_free={r['ilp_D_free_time_s']:.3f}s  "
            f"DP1={'OK' if r['U_match_DP1'] else 'FAIL'}"
        )
    n = len(rows)
    match = sum(1 for r in rows if r["U_match_DP1"])
    mean_tA = sum(r["ilp_A_time_s"] for r in rows) / n
    mean_tF = sum(r["ilp_D_free_time_s"] for r in rows) / n
    mean_tZ = sum(r["ilp_D_forced_time_s"] for r in rows) / n
    mean_rho = sum(r["rho"] for r in rows) / n
    print(
        f"Summary n={args.n_nodes}: DP1_match={match}/{n}  "
        f"rho_mean={mean_rho:.4f}  "
        f"mean_t_A={mean_tA:.4f}s mean_t_D_free={mean_tF:.4f}s "
        f"mean_t_D_forced={mean_tZ:.4f}s"
    )
    print(f"Wrote {out}")

if __name__ == "__main__":
    main()
