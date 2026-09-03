"""
run_hard_ablation.py — T2c: hard conflict gadget + seed sweep.

So sánh Greedy / Random+Repair / QAOA+Repair trên instance có local-optima
trap (H vs A+B). Không cần NetSquid.

Chạy:
    python3 examples/run_hard_ablation.py --n-trials 50
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

from qos_sim.topology_gen import generate_hard_conflict_instance, hard_instance_to_qubo
from qos_sim.qaoa_numpy import run_qaoa
from qos_sim.gwo_optimizer import optimize as gwo_optimize
from qos_sim.repair import repair, verify_feasible
from qos_sim.qubo_builder import served_utility
from qos_sim.baseline import greedy_schedule, random_init_and_repair, exact_solver_pulp
from qos_sim.dvqe_partition import partition_by_weakest_edge, marginals_from_probabilities

P_DEPTH = 3
MAX_QUBITS = 10


def run_qaoa_pipeline(qubo, requests, conflicts, rng):
    n = len(qubo.variables)
    if n > MAX_QUBITS:
        groups = partition_by_weakest_edge(qubo.Q, MAX_QUBITS)
    else:
        groups = [list(range(n))]

    x_raw = np.zeros(n)
    for group in groups:
        sub_Q = qubo.Q[np.ix_(group, group)]

        def fitness(params, sub_Q=sub_Q):
            g, b = params[:P_DEPTH], params[P_DEPTH:]
            return run_qaoa(sub_Q, float(g.mean()), float(b.mean()), p=P_DEPTH).expectation

        gwo_r = gwo_optimize(fitness, dim=2 * P_DEPTH, n_wolves=20, max_iter=50, rng=rng)
        g_b = float(gwo_r.best_position[:P_DEPTH].mean())
        b_b = float(gwo_r.best_position[P_DEPTH:].mean())
        qr = run_qaoa(sub_Q, g_b, b_b, p=P_DEPTH)
        best = int(np.argmax(qr.probabilities))
        for li, gi in enumerate(group):
            x_raw[gi] = (best >> li) & 1

    raw_u = served_utility(qubo, x_raw)
    raw_feas = verify_feasible(x_raw, conflicts)
    x = repair(qubo, x_raw, requests, conflicts)
    return x, served_utility(qubo, x), verify_feasible(x, conflicts), raw_u, raw_feas


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-trials", type=int, default=50)
    parser.add_argument("--n-extra", type=int, default=8)
    parser.add_argument("--base-seed", type=int, default=42)
    args = parser.parse_args()

    rows = []
    t0 = time.perf_counter()
    for i in range(args.n_trials):
        seed = args.base_seed + i
        requests, variables, conflicts, weights = generate_hard_conflict_instance(
            n_extra=args.n_extra, seed=seed
        )
        qubo = hard_instance_to_qubo(requests, variables, conflicts)
        rng = np.random.default_rng(seed)

        gr = greedy_schedule(qubo, requests, conflicts)
        rr = random_init_and_repair(qubo, requests, conflicts, seed=seed)
        x_q, q_u, q_feas, raw_u, raw_feas = run_qaoa_pipeline(
            qubo, requests, conflicts, rng
        )
        exact = exact_solver_pulp(qubo, requests, conflicts, time_limit_s=30.0)
        opt_u = exact.served_utility

        trap_gap = (weights["A"] + weights["B"]) - weights["H"]
        gap_q = (opt_u - q_u) / opt_u * 100.0 if opt_u > 1e-9 else 0.0
        gap_g = (opt_u - gr.served_utility) / opt_u * 100.0 if opt_u > 1e-9 else 0.0
        row = {
            "seed": seed,
            "n_vars": len(variables),
            "n_conflicts": len(conflicts),
            "w_H": weights["H"],
            "w_A": weights["A"],
            "w_B": weights["B"],
            "trap_gap": trap_gap,
            "greedy_U": gr.served_utility,
            "random_repair_U": rr.served_utility,
            "qaoa_repair_U": q_u,
            "raw_qaoa_U": raw_u,
            "exact_U": opt_u,
            "exact_runtime_s": exact.runtime_s,
            "gap_qaoa_pct": gap_q,
            "gap_greedy_pct": gap_g,
            "greedy_eq_opt": abs(gr.served_utility - opt_u) < 1e-6,
            "qaoa_eq_opt": abs(q_u - opt_u) < 1e-6,
            "random_eq_opt": abs(rr.served_utility - opt_u) < 1e-6,
            "qaoa_beats_greedy": q_u > gr.served_utility + 1e-9,
            "greedy_took_H": None,
        }
        h_idx = next(i for i, v in enumerate(variables) if v.request_id == "H")
        row["greedy_took_H"] = bool(gr.x[h_idx] > 0.5)
        rows.append(row)
        print(
            f"  [{i+1}/{args.n_trials}] seed={seed}  "
            f"G={gr.served_utility:.1f}  R={rr.served_utility:.1f}  "
            f"Q={q_u:.1f}  Exact={opt_u:.1f}  "
            f"gapQ={gap_q:.1f}% gapG={gap_g:.1f}%  "
            f"Q>G:{row['qaoa_beats_greedy']}",
            flush=True,
        )


    out = "hard_ablation_raw.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    def ms(key):
        vals = [float(r[key]) for r in rows]
        return statistics.mean(vals), statistics.stdev(vals) if len(vals) > 1 else 0.0

    print(f"\n=== Hard ablation (PuLP+CBC exact)  N={len(rows)}  n_extra={args.n_extra} ===")
    for label, key in [
        ("Greedy", "greedy_U"),
        ("Random+Repair", "random_repair_U"),
        ("QAOA+Repair", "qaoa_repair_U"),
        ("Exact (CBC)", "exact_U"),
        ("Raw QAOA", "raw_qaoa_U"),
    ]:
        m, s = ms(key)
        print(f"  {label:16s}  {m:.2f} ± {s:.2f}")

    n = len(rows)
    print(f"  Greedy == Exact     : {sum(1 for r in rows if r['greedy_eq_opt'])}/{n}")
    print(f"  Random == Exact     : {sum(1 for r in rows if r['random_eq_opt'])}/{n}")
    print(f"  QAOA+Repair == Exact: {sum(1 for r in rows if r['qaoa_eq_opt'])}/{n}")
    print(f"  QAOA beats Greedy   : {sum(1 for r in rows if r['qaoa_beats_greedy'])}/{n}")
    print(f"  Greedy took H       : {sum(1 for r in rows if r['greedy_took_H'])}/{n}")
    print(f"  mean gap QAOA (%)   : {ms('gap_qaoa_pct')[0]:.2f}")
    print(f"  mean gap Greedy (%) : {ms('gap_greedy_pct')[0]:.2f}")
    print(f"  mean exact runtime  : {ms('exact_runtime_s')[0]:.4f}s")
    print(f"  mean trap_gap       : {ms('trap_gap')[0]:.2f}")
    print(f"  elapsed {time.perf_counter()-t0:.1f}s  → {out}")



if __name__ == "__main__":
    main()
