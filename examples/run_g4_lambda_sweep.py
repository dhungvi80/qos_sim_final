"""
G4: lambda sweep (lambda = margin * lambda*).

Margins {1.0, 1.1, 1.5, 2.0} x n_trials at given scale.
  - ILP on A (constraint form)
  - SA on QUBO energy -> raw feasibility + U after repair
  - 1-flip check: infeasible y with H(y) <= H(x_ILP)?

  python3 -u examples/run_g4_lambda_sweep.py --n-nodes 25 --n-trials 20
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
from qos_sim.qubo_builder import (
    build_qubo, conflict_pairs_from_paths, served_utility, objective_value,
)
from qos_sim.baseline import exact_solver_highs, simulated_annealing
from qos_sim.repair import repair, verify_feasible

F_MIN = 0.95
MARGINS = (1.0, 1.1, 1.5, 2.0)


def _has_improving_infeasible_flip(qubo, x, conflicts) -> bool:
    Hx = objective_value(qubo, x)
    n = len(x)
    for i in range(n):
        y = x.copy()
        y[i] = 1.0 - y[i]
        if verify_feasible(y, conflicts):
            continue
        if objective_value(qubo, y) <= Hx + 1e-9:
            return True
    return False


def one_seed(n_nodes: int, seed: int, margins, time_limit_s: float):
    ns.set_qstate_formalism(QFormalism.DM)
    node_ids, links, graph = generate_topology(n_nodes, seed=seed)
    requests, paths, time_slots, path_resources, valid_paths = generate_requests_and_paths(
        graph, n_requests=n_nodes, k_paths=2, n_timeslots=3, seed=seed
    )
    network = build_topology(node_ids, links)
    trace = run_trace(network, paths, time_slots, link_params=links)
    telemetry = NetSquidTelemetry(trace)
    active, stats = prune(
        requests, paths, time_slots, telemetry,
        fidelity_threshold=F_MIN, valid_paths=valid_paths, collision_pruning=False,
    )
    conflicts = conflict_pairs_from_paths(active, path_resources)
    qubo1 = build_qubo(active, requests, conflicts, lambda_margin=1.0)
    exact = exact_solver_highs(qubo1, requests, conflicts, time_limit_s=time_limit_s)

    rows = []
    for m in margins:
        qubo = build_qubo(active, requests, conflicts, lambda_margin=m)
        x_ilp = exact.x
        H_ilp = objective_value(qubo, x_ilp) if exact.optimal else float("nan")
        flip_bad = (
            _has_improving_infeasible_flip(qubo, x_ilp, conflicts)
            if exact.optimal else None
        )
        sa = simulated_annealing(qubo, conflicts, n_iters=2000, seed=seed)
        sa_feas = verify_feasible(sa.x, conflicts)
        sa_rep = repair(qubo, sa.x, requests, conflicts)
        sa_rep_u = served_utility(qubo, sa_rep)
        rows.append({
            "n_nodes": n_nodes,
            "trial_seed": seed,
            "size_A": len(active),
            "lambda_star": qubo.lambda_star,
            "lambda_margin": m,
            "lambda_used": qubo.lambda_used,
            "ilp_U": exact.served_utility,
            "ilp_optimal": bool(exact.optimal),
            "H_ilp": H_ilp,
            "improving_infeas_flip": flip_bad,
            "sa_raw_feasible": bool(sa_feas),
            "sa_raw_U": served_utility(qubo, sa.x),
            "sa_repair_U": sa_rep_u,
            "gap_sa_repair_pct": (
                100.0 * (1.0 - sa_rep_u / exact.served_utility)
                if exact.optimal and exact.served_utility > 0 else float("nan")
            ),
        })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-nodes", type=int, default=25)
    ap.add_argument("--n-trials", type=int, default=20)
    ap.add_argument("--seed0", type=int, default=42)
    ap.add_argument("--time-limit", type=float, default=30.0)
    args = ap.parse_args()

    out = Path(__file__).resolve().parents[1] / f"g4_lambda_sweep_{args.n_nodes}.csv"
    fieldnames = None
    all_rows = []
    print(f"=== G4 lambda sweep  n={args.n_nodes} trials={args.n_trials} "
          f"margins={list(MARGINS)} ===")
    for i in range(args.n_trials):
        seed = args.seed0 + i
        rows = one_seed(args.n_nodes, seed, MARGINS, args.time_limit)
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
        by_m = {r["lambda_margin"]: r for r in rows}
        bits = "  ".join(
            f"m={m}:sa_feas={int(by_m[m]['sa_raw_feasible'])}"
            f"/flip={by_m[m]['improving_infeas_flip']}"
            f"/gap={by_m[m]['gap_sa_repair_pct']:.2f}%"
            for m in MARGINS
        )
        print(f"  seed={seed:4d} |A|={rows[0]['size_A']:3d} λ*={rows[0]['lambda_star']:.2f}  {bits}")

    print("--- Summary by margin ---")
    for m in MARGINS:
        sub = [r for r in all_rows if r["lambda_margin"] == m]
        n = len(sub)
        sa_f = sum(1 for r in sub if r["sa_raw_feasible"])
        flip = sum(1 for r in sub if r["improving_infeas_flip"])
        gaps = [r["gap_sa_repair_pct"] for r in sub if r["gap_sa_repair_pct"] == r["gap_sa_repair_pct"]]
        mean_gap = sum(gaps) / len(gaps) if gaps else float("nan")
        print(
            f"  margin={m}: sa_raw_feas={sa_f}/{n}  "
            f"improving_infeas_flip={flip}/{n}  "
            f"mean_gap_sa_repair={mean_gap:.3f}%"
        )
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
