"""
D4-B: FN rates under telemetry noise + lambda* vs max request weight.
  python3 -u examples/run_d4b_fn_lambda.py --n-nodes 25 --n-trials 20
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-nodes", type=int, default=25)
    ap.add_argument("--n-trials", type=int, default=20)
    ap.add_argument("--seed0", type=int, default=42)
    args = ap.parse_args()
    out = Path(__file__).resolve().parents[1] / f"d4b_fn_lambda_{args.n_nodes}.csv"
    fieldnames, all_rows = None, []
    print(f"=== D4-B FN + lambda*  n={args.n_nodes} trials={args.n_trials} ===")
    for i in range(args.n_trials):
        seed = args.seed0 + i
        ns.set_qstate_formalism(QFormalism.DM)
        node_ids, links, graph = generate_topology(args.n_nodes, seed=seed)
        requests, paths, time_slots, path_resources, valid_paths = generate_requests_and_paths(
            graph, n_requests=args.n_nodes, k_paths=2, n_timeslots=3, seed=seed
        )
        network = build_topology(node_ids, links)
        trace = run_trace(network, paths, time_slots, link_params=links)
        base = NetSquidTelemetry(trace)
        active0, st0 = prune(
            requests, paths, time_slots, base,
            fidelity_threshold=F_MIN, valid_paths=valid_paths, collision_pruning=False,
        )
        K0 = keys(active0)
        conflicts = conflict_pairs_from_paths(active0, path_resources)
        qubo = build_qubo(active0, requests, conflicts, lambda_margin=1.0)
        ws = []
        for r in requests:
            w = getattr(r, "weight", None)
            if w is None:
                w = getattr(r, "utility", 1.0)
            ws.append(float(w))
        max_w = max(ws) if ws else float("nan")
        lam = float(qubo.lambda_star)
        line = f"  seed={seed:4d} lambda*={lam:.3f} max_w={max_w:.3f} ratio={lam/max_w:.3f}"
        for eps in EPS_LIST:
            rng = np.random.default_rng(seed * 1000 + int(eps * 1000))
            tel = NoisyTelemetry(base, eps, rng) if eps > 0 else base
            active_e, st_e = prune(
                requests, paths, time_slots, tel,
                fidelity_threshold=F_MIN, valid_paths=valid_paths, collision_pruning=False,
            )
            Ke = keys(active_e)
            fp, fn = Ke - K0, K0 - Ke
            rec = {
                "n_nodes": args.n_nodes,
                "trial_seed": seed,
                "eps": eps,
                "size_A0": len(active0),
                "size_A_eps": len(active_e),
                "n_fp": len(fp),
                "n_fn": len(fn),
                "fp_rate": len(fp) / len(Ke) if Ke else 0.0,
                "fn_rate": len(fn) / len(K0) if K0 else 0.0,
                "lambda_star": lam,
                "max_w": max_w,
                "lam_over_max_w": lam / max_w if max_w else float("nan"),
            }
            all_rows.append(rec)
            if fieldnames is None:
                fieldnames = list(rec.keys())
                with out.open("w", newline="") as f:
                    w = csv.DictWriter(f, fieldnames=fieldnames)
                    w.writeheader()
                    w.writerow(rec)
            else:
                with out.open("a", newline="") as f:
                    csv.DictWriter(f, fieldnames=fieldnames).writerow(rec)
            line += f"  eps={eps}:fp={len(fp)} fn={len(fn)}"
        print(line)
    print("--- Summary lambda* ---")
    sub0 = [r for r in all_rows if r["eps"] == 0.0]
    n = len(sub0)
    print(
        f"  mean lambda*={sum(r['lambda_star'] for r in sub0)/n:.3f}  "
        f"mean max_w={sum(r['max_w'] for r in sub0)/n:.3f}  "
        f"mean ratio={sum(r['lam_over_max_w'] for r in sub0)/n:.3f}"
    )
    print("--- Summary FN/FP by eps ---")
    for e in EPS_LIST:
        sub = [r for r in all_rows if r["eps"] == e]
        n = len(sub)
        print(
            f"  eps={e}: mean_fp_rate={sum(r['fp_rate'] for r in sub)/n:.3f}  "
            f"mean_fn_rate={sum(r['fn_rate'] for r in sub)/n:.3f}  "
            f"mean_fn={sum(r['n_fn'] for r in sub)/n:.1f}"
        )
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
