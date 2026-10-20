"""
D3-B: weakest-edge vs request-grouped partition boundary fractions.
  python3 -u examples/run_d3b_partition_compare.py --n-nodes 25 --n-trials 20
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
    total = boundary = 0
    for i in range(n):
        for j in range(i + 1, n):
            if abs(Q[i, j]) < 1e-15 and abs(Q[j, i]) < 1e-15:
                continue
            total += 1
            if assign[i] != assign[j]:
                boundary += 1
    return total, boundary, (boundary / total if total else 0.0)


def partition_by_request(variables, n_max=N_MAX):
    by_req = {}
    for i, v in enumerate(variables):
        by_req.setdefault(v.request_id, []).append(i)
    req_groups = sorted(by_req.values(), key=len, reverse=True)
    bins, sizes = [], []
    for g in req_groups:
        if len(g) > n_max:
            for k in range(0, len(g), n_max):
                bins.append(g[k:k + n_max])
                sizes.append(len(bins[-1]))
            continue
        placed = False
        for bi, b in enumerate(bins):
            if sizes[bi] + len(g) <= n_max:
                b.extend(g)
                sizes[bi] += len(g)
                placed = True
                break
        if not placed:
            bins.append(list(g))
            sizes.append(len(g))
    return bins


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-nodes", type=int, default=25)
    ap.add_argument("--n-trials", type=int, default=20)
    ap.add_argument("--seed0", type=int, default=42)
    args = ap.parse_args()
    out = Path(__file__).resolve().parents[1] / f"d3b_partition_compare_{args.n_nodes}.csv"
    fieldnames, rows = None, []
    print(f"=== D3-B partition compare  n={args.n_nodes} trials={args.n_trials} ===")
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
        g_weak = partition_by_weakest_edge(qubo.Q, N_MAX) if n > N_MAX else [list(range(n))]
        g_req = partition_by_request(active, N_MAX)
        _, _, r_w = boundary_ratio(qubo.Q, g_weak)
        _, _, r_r = boundary_ratio(qubo.Q, g_req)
        rec = {
            "n_nodes": args.n_nodes, "trial_seed": seed, "size_A": n,
            "M_weak": len(g_weak), "bnd_weak": r_w,
            "M_req": len(g_req), "bnd_req": r_r,
            "bnd_diff": r_w - r_r,
        }
        rows.append(rec)
        if fieldnames is None:
            fieldnames = list(rec.keys())
            with out.open("w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=fieldnames)
                w.writeheader(); w.writerow(rec)
        else:
            with out.open("a", newline="") as f:
                csv.DictWriter(f, fieldnames=fieldnames).writerow(rec)
        print(
            f"  seed={seed:4d} |A|={n}  "
            f"weak: M={len(g_weak)} bnd={r_w:.3f}  "
            f"req: M={len(g_req)} bnd={r_r:.3f}  "
            f"diff={r_w-r_r:+.3f}"
        )
    n = len(rows)
    print("--- Summary ---")
    print(
        f"  mean bnd_weak={sum(r['bnd_weak'] for r in rows)/n:.3f}  "
        f"mean bnd_req={sum(r['bnd_req'] for r in rows)/n:.3f}  "
        f"mean M_weak={sum(r['M_weak'] for r in rows)/n:.1f}  "
        f"mean M_req={sum(r['M_req'] for r in rows)/n:.1f}"
    )
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
