"""
D1-B light: partition statistics on A (weakest-edge cut).
Reports |A|, M, mean partition size, boundary/total edge ratio.

  python3 -u examples/run_d1b_partition_stats.py --n-nodes 25 --n-trials 20
"""
from __future__ import annotations
import argparse, csv, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import netsquid as ns
from netsquid.qubits import QFormalism
from qos_sim.topology_gen import generate_topology, generate_requests_and_paths
from qos_sim.pruning import prune
from qos_sim.simulator import NetSquidTelemetry, build_topology, run_trace
from qos_sim.qubo_builder import build_qubo, conflict_pairs_from_paths
from qos_sim.dvqe_partition import partition_by_weakest_edge

F_MIN = 0.95
N_MAX = 10


def boundary_ratio(Q, groups):
    n = Q.shape[0]
    assign = np.empty(n, dtype=int)
    for gi, g in enumerate(groups):
        for i in g:
            assign[i] = gi
    total = 0
    boundary = 0
    for i in range(n):
        for j in range(i + 1, n):
            if abs(Q[i, j]) < 1e-15 and abs(Q[j, i]) < 1e-15:
                continue
            total += 1
            if assign[i] != assign[j]:
                boundary += 1
    return total, boundary, (boundary / total if total else 0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-nodes", type=int, default=25)
    ap.add_argument("--n-trials", type=int, default=20)
    ap.add_argument("--seed0", type=int, default=42)
    args = ap.parse_args()
    out = Path(__file__).resolve().parents[1] / f"d1b_partition_stats_{args.n_nodes}.csv"
    fieldnames, rows = None, []
    print(f"=== D1-B partition stats  n={args.n_nodes} trials={args.n_trials} n_max={N_MAX} ===")
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
        n = len(active)
        if n > N_MAX:
            groups = partition_by_weakest_edge(qubo.Q, N_MAX)
        else:
            groups = [list(range(n))]
        sizes = [len(g) for g in groups]
        tot, bdy, ratio = boundary_ratio(qubo.Q, groups)
        rec = {
            "n_nodes": args.n_nodes, "trial_seed": seed,
            "size_A": n, "M": len(groups),
            "mean_part": float(np.mean(sizes)), "max_part": max(sizes),
            "n_edges": tot, "n_boundary": bdy, "boundary_frac": ratio,
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
        print(
            f"  seed={seed:4d} |A|={n} M={len(groups)} "
            f"mean_part={np.mean(sizes):.1f} max={max(sizes)} "
            f"bnd_frac={ratio:.3f}"
        )
    n = len(rows)
    print("--- Summary ---")
    print(
        f"  mean |A|={sum(r['size_A'] for r in rows)/n:.1f}  "
        f"mean M={sum(r['M'] for r in rows)/n:.1f}  "
        f"mean bnd_frac={sum(r['boundary_frac'] for r in rows)/n:.3f}"
    )
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
