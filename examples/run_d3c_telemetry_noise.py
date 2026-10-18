"""
D3-C: telemetry fidelity noise robustness (unary pruning).

eps in {0, 0.05, 0.10}: perturb F' = clip(F*(1+u), 0, 1), u~Uniform(-eps,eps)
then unary prune and solve ILP/Greedy on A.

  python3 -u examples/run_d3c_telemetry_noise.py --n-nodes 25 --n-trials 20
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
from qos_sim.baseline import exact_solver_highs, greedy_schedule

F_MIN = 0.95
EPS_LIST = (0.0, 0.05, 0.10)


class NoisyTelemetry:
    def __init__(self, base, eps: float, rng: np.random.Generator):
        self.base = base
        self.eps = float(eps)
        self.rng = rng
        self._cache = {}

    def fidelity(self, path, t: int) -> float:
        key = (getattr(path, "id", id(path)), t)
        if key not in self._cache:
            f = float(self.base.fidelity(path, t))
            if self.eps > 0:
                u = self.rng.uniform(-self.eps, self.eps)
                f = float(np.clip(f * (1.0 + u), 0.0, 1.0))
            self._cache[key] = f
        return self._cache[key]

    def is_available(self, path, t: int) -> bool:
        return self.base.is_available(path, t)


def one_seed(n_nodes: int, seed: int, eps_list, time_limit_s: float):
    ns.set_qstate_formalism(QFormalism.DM)
    node_ids, links, graph = generate_topology(n_nodes, seed=seed)
    requests, paths, time_slots, path_resources, valid_paths = generate_requests_and_paths(
        graph, n_requests=n_nodes, k_paths=2, n_timeslots=3, seed=seed
    )
    network = build_topology(node_ids, links)
    trace = run_trace(network, paths, time_slots, link_params=links)
    base_tel = NetSquidTelemetry(trace)
    rows = []
    for eps in eps_list:
        rng = np.random.default_rng(seed * 1000 + int(eps * 1000))
        tel = NoisyTelemetry(base_tel, eps, rng) if eps > 0 else base_tel
        active, stats = prune(
            requests, paths, time_slots, tel,
            fidelity_threshold=F_MIN, valid_paths=valid_paths, collision_pruning=False,
        )
        if len(active) == 0:
            rows.append({
                "n_nodes": n_nodes, "trial_seed": seed, "eps": eps,
                "size_D": stats.size_D, "size_A": 0, "rho": 1.0,
                "ilp_U": float("nan"), "greedy_U": float("nan"),
                "ilp_optimal": False,
            })
            continue
        conflicts = conflict_pairs_from_paths(active, path_resources)
        qubo = build_qubo(active, requests, conflicts, lambda_margin=1.0)
        exact = exact_solver_highs(qubo, requests, conflicts, time_limit_s=time_limit_s)
        greedy = greedy_schedule(qubo, requests, conflicts)
        rows.append({
            "n_nodes": n_nodes, "trial_seed": seed, "eps": eps,
            "size_D": stats.size_D, "size_A": len(active),
            "rho": 1.0 - len(active) / stats.size_D if stats.size_D else 0.0,
            "ilp_U": exact.served_utility,
            "greedy_U": greedy.served_utility,
            "ilp_optimal": bool(exact.optimal),
        })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-nodes", type=int, default=25)
    ap.add_argument("--n-trials", type=int, default=20)
    ap.add_argument("--seed0", type=int, default=42)
    ap.add_argument("--time-limit", type=float, default=30.0)
    args = ap.parse_args()
    out = Path(__file__).resolve().parents[1] / f"d3c_telemetry_noise_{args.n_nodes}.csv"
    fieldnames = None
    all_rows = []
    print(f"=== D3-C telemetry noise  n={args.n_nodes} trials={args.n_trials} eps={list(EPS_LIST)} ===")
    for i in range(args.n_trials):
        seed = args.seed0 + i
        rows = one_seed(args.n_nodes, seed, EPS_LIST, args.time_limit)
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
        by = {r["eps"]: r for r in rows}
        bits = "  ".join(
            f"eps={e}:|A|={by[e]['size_A']} rho={by[e]['rho']:.3f} U={by[e]['ilp_U']:.2f}"
            for e in EPS_LIST
        )
        print(f"  seed={seed:4d}  {bits}")

    print("--- Summary by eps ---")
    for e in EPS_LIST:
        sub = [r for r in all_rows if r["eps"] == e]
        n = len(sub)
        mean_A = sum(r["size_A"] for r in sub) / n
        mean_rho = sum(r["rho"] for r in sub) / n
        us = [r["ilp_U"] for r in sub if r["ilp_U"] == r["ilp_U"]]
        mean_u = sum(us) / len(us) if us else float("nan")
        print(f"  eps={e}: mean_|A|={mean_A:.1f}  mean_rho={mean_rho:.4f}  mean_ILP_U={mean_u:.3f}")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
