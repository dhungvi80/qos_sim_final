"""
M6: pruning vs full space — |A|, estimated DVQE partitions, conflict split.
  python3 -u examples/run_m6_pruning_partitions.py --n-nodes 25 --n-trials 20
"""
from __future__ import annotations
import argparse, csv, sys, math
from pathlib import Path
from itertools import combinations
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import netsquid as ns
from netsquid.qubits import QFormalism
from qos_sim.topology_gen import generate_topology, generate_requests_and_paths
from qos_sim.pruning import prune, Variable
from qos_sim.simulator import NetSquidTelemetry, build_topology, run_trace
from qos_sim.qubo_builder import conflict_pairs_from_paths

F_MIN = 0.95
N_PART = 10


def all_vars(requests, paths, time_slots, valid_paths):
    paths_by_id = {p.id: p for p in paths}
    out = []
    for r in requests:
        if valid_paths is None:
            cand = list(paths)
        else:
            cand = [paths_by_id[pid] for pid in sorted(valid_paths.get(r.id, ()))
                    if pid in paths_by_id]
        for p in cand:
            for t in time_slots:
                out.append(Variable(r.id, p.id, t))
    return out


def conflict_split(variables, path_resources):
    by_req = {}
    for i, v in enumerate(variables):
        by_req.setdefault(v.request_id, []).append(i)
    uniq = sum(len(list(combinations(idxs, 2))) for idxs in by_req.values())
    pairs = conflict_pairs_from_paths(variables, path_resources)
    res = max(0, len(pairs) - uniq)
    return len(pairs), uniq, res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-nodes", type=int, default=25)
    ap.add_argument("--n-trials", type=int, default=20)
    ap.add_argument("--seed0", type=int, default=42)
    args = ap.parse_args()
    out = Path(__file__).resolve().parents[1] / f"m6_pruning_partitions_{args.n_nodes}.csv"
    fieldnames, rows = None, []
    print(f"=== M6 pruning partitions  n={args.n_nodes} trials={args.n_trials} ===")
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
        full = all_vars(requests, paths, time_slots, valid_paths)
        cA, uA, rA = conflict_split(active, path_resources)
        cD, uD, rD = conflict_split(full, path_resources)
        rec = {
            "n_nodes": args.n_nodes, "trial_seed": seed,
            "size_D": len(full), "size_A": len(active),
            "M_D_est": math.ceil(len(full) / N_PART) if full else 0,
            "M_A_est": math.ceil(len(active) / N_PART) if active else 0,
            "conflicts_A": cA, "uniq_A": uA, "resource_A": rA,
            "conflicts_D": cD, "uniq_D": uD, "resource_D": rD,
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
            f"  seed={seed:4d} |D|={rec['size_D']} |A|={rec['size_A']}  "
            f"M_D~{rec['M_D_est']} M_A~{rec['M_A_est']}  "
            f"C_A={cA} (uniq={uA}, res={rA})"
        )
    n = len(rows)
    print("--- Summary ---")
    print(
        f"  mean |A|={sum(r['size_A'] for r in rows)/n:.1f}  "
        f"mean M_A={sum(r['M_A_est'] for r in rows)/n:.1f}  "
        f"mean M_D={sum(r['M_D_est'] for r in rows)/n:.1f}  "
        f"mean uniq_A frac={sum(r['uniq_A']/max(r['conflicts_A'],1) for r in rows)/n:.3f}"
    )
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
