"""
D4-B: telemetry noise evaluated against TRUE fidelity.
  python3 -u examples/run_d4b_noise_true_F.py --n-nodes 25 --n-trials 20
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
from qos_sim.baseline import exact_solver_highs

F_MIN = 0.95
EPS_LIST = (0.0, 0.05, 0.10)


class NoisyTelemetry:
    def __init__(self, base, eps, rng):
        self.base, self.eps, self.rng = base, float(eps), rng
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


def keys(active):
    return {v.key() for v in active}


def filter_active(active, keyset):
    return [v for v in active if v.key() in keyset]


def ilp_on(active, requests, path_resources, time_limit_s):
    if not active:
        return float("nan"), False
    conflicts = conflict_pairs_from_paths(active, path_resources)
    qubo = build_qubo(active, requests, conflicts, lambda_margin=1.0)
    ex = exact_solver_highs(qubo, requests, conflicts, time_limit_s=time_limit_s)
    return float(ex.served_utility), bool(ex.optimal)


def one_seed(n_nodes, seed, time_limit_s):
    ns.set_qstate_formalism(QFormalism.DM)
    node_ids, links, graph = generate_topology(n_nodes, seed=seed)
    requests, paths, time_slots, path_resources, valid_paths = generate_requests_and_paths(
        graph, n_requests=n_nodes, k_paths=2, n_timeslots=3, seed=seed
    )
    network = build_topology(node_ids, links)
    trace = run_trace(network, paths, time_slots, link_params=links)
    base = NetSquidTelemetry(trace)
    active0, st0 = prune(
        requests, paths, time_slots, base,
        fidelity_threshold=F_MIN, valid_paths=valid_paths, collision_pruning=False,
    )
    K0 = keys(active0)
    u0, opt0 = ilp_on(active0, requests, path_resources, time_limit_s)
    rows = []
    for eps in EPS_LIST:
        rng = np.random.default_rng(seed * 1000 + int(eps * 1000))
        tel = NoisyTelemetry(base, eps, rng) if eps > 0 else base
        active_e, st_e = prune(
            requests, paths, time_slots, tel,
            fidelity_threshold=F_MIN, valid_paths=valid_paths, collision_pruning=False,
        )
        Ke = keys(active_e)
        fp, fn = Ke - K0, K0 - Ke
        inter = Ke & K0
        active_true = filter_active(active_e, inter)
        u_true, opt_t = ilp_on(active_true, requests, path_resources, time_limit_s)
        u_noisy, opt_n = ilp_on(active_e, requests, path_resources, time_limit_s)
        rows.append({
            "n_nodes": n_nodes, "trial_seed": seed, "eps": eps,
            "size_D": st0.size_D, "size_A0": len(active0), "size_A_eps": len(active_e),
            "n_false_pos": len(fp), "n_false_neg": len(fn), "n_intersection": len(inter),
            "fp_rate": len(fp) / len(Ke) if Ke else 0.0,
            "ilp_U_A0": u0, "ilp_U_Aeps_true": u_true, "ilp_U_Aeps_raw": u_noisy,
            "opt_A0": opt0, "opt_true": opt_t, "opt_raw": opt_n,
        })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-nodes", type=int, default=25)
    ap.add_argument("--n-trials", type=int, default=20)
    ap.add_argument("--seed0", type=int, default=42)
    ap.add_argument("--time-limit", type=float, default=30.0)
    args = ap.parse_args()
    out = Path(__file__).resolve().parents[1] / f"d4b_noise_true_F_{args.n_nodes}.csv"
    fieldnames, all_rows = None, []
    print(f"=== D4-B noise true-F  n={args.n_nodes} trials={args.n_trials} ===")
    for i in range(args.n_trials):
        seed = args.seed0 + i
        rows = one_seed(args.n_nodes, seed, args.time_limit)
        for r in rows:
            all_rows.append(r)
            if fieldnames is None:
                fieldnames = list(r.keys())
                with out.open("w", newline="") as f:
                    w = csv.DictWriter(f, fieldnames=fieldnames)
                    w.writeheader(); w.writerow(r)
            else:
                with out.open("a", newline="") as f:
                    csv.DictWriter(f, fieldnames=fieldnames).writerow(r)
        by = {r["eps"]: r for r in rows}
        print(
            f"  seed={seed:4d}  "
            + "  ".join(
                f"eps={e}:|A|={by[e]['size_A_eps']} fp={by[e]['n_false_pos']} "
                f"U_true={by[e]['ilp_U_Aeps_true']:.1f} U_raw={by[e]['ilp_U_Aeps_raw']:.1f}"
                for e in EPS_LIST
            )
        )
    print("--- Summary by eps ---")
    for e in EPS_LIST:
        sub = [r for r in all_rows if r["eps"] == e]
        n = len(sub)
        def mean(k):
            return sum(r[k] for r in sub) / n
        print(
            f"  eps={e}: |A|={mean('size_A_eps'):.1f}  fp_rate={mean('fp_rate'):.3f}  "
            f"U_A0={mean('ilp_U_A0'):.2f}  U_true={mean('ilp_U_Aeps_true'):.2f}  "
            f"U_raw={mean('ilp_U_Aeps_raw'):.2f}"
        )
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
