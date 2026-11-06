"""
M3: Per-partition QAOA mode vs exhaustive local optimum (2^{n_part}).
Uses hop-based synthetic telemetry (same F_min=0.95, T2-order depolar model)
when NetSquid is unavailable; QUBO/partition/QAOA path matches the paper.

  python3 -u examples/run_m3_partition_exact.py --n-nodes 25 --n-trials 20
"""
from __future__ import annotations
import argparse, csv, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from qos_sim.topology_gen import generate_topology, generate_requests_and_paths
from qos_sim.pruning import NetworkTelemetry, PathConfig, prune
from qos_sim.qubo_builder import build_qubo, conflict_pairs_from_paths
from qos_sim.dvqe_partition import partition_by_weakest_edge
from qos_sim.qaoa_numpy import run_qaoa
from qos_sim.gwo_optimizer import optimize as gwo_optimize

F_MIN = 0.95
N_MAX = 10
P_DEPTH = 3


class HopFidelityTelemetry(NetworkTelemetry):
    """Physical-style synthetic telemetry without NetSquid.

    End-to-end fidelity decays with hop count via a fixed per-hop factor
    consistent with the paper's T2~10 ms / few-km link regime (order-of-
    magnitude product model). Availability is always true so that unary
    pruning is driven by fidelity alone (matches collision_pruning=False
    main runs where availability rarely binds).
    """

    def __init__(self, per_hop: float = 0.97, seed: int = 0):
        self.per_hop = per_hop
        self.rng = np.random.default_rng(seed)

    def fidelity(self, path: PathConfig, t: int) -> float:
        n_hops = max(1, len(path.resources))
        # mild slot dependence + hop decay
        base = self.per_hop ** n_hops
        jitter = 1.0 - 0.02 * ((t + hash(path.id) % 7) % 3)
        return float(np.clip(base * jitter, 0.0, 1.0))

    def is_available(self, path: PathConfig, t: int) -> bool:
        return True

    def has_collision(self, path, t, reserved):
        for resource in path.resources:
            holders = reserved.get((resource, t), set())
            if holders and path.id not in holders:
                return True
        return False


def exact_min_bitstring(Q: np.ndarray):
    n = Q.shape[0]
    dim = 1 << n
    states = np.arange(dim)[:, None]
    bits = ((states >> np.arange(n)) & 1).astype(float)
    energies = np.einsum("si,ij,sj->s", bits, Q, bits)
    best = int(np.argmin(energies))
    return best, float(energies[best]), n


def state_energy(Q: np.ndarray, state: int) -> float:
    n = Q.shape[0]
    bits = np.array([(state >> k) & 1 for k in range(n)], dtype=float)
    return float(bits @ Q @ bits)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-nodes", type=int, default=25)
    ap.add_argument("--n-trials", type=int, default=20)
    ap.add_argument("--seed0", type=int, default=42)
    ap.add_argument("--n-wolves", type=int, default=30)
    ap.add_argument("--max-iter", type=int, default=80)
    args = ap.parse_args()

    out = Path(__file__).resolve().parents[1] / "results" / "m3_partition_exact.csv"
    out.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    t_all = time.time()
    print(
        f"=== M3 exact-vs-mode  n={args.n_nodes} trials={args.n_trials} "
        f"GWO={args.n_wolves}x{args.max_iter} p={P_DEPTH} n_max={N_MAX} ===",
        flush=True,
    )

    for t in range(args.n_trials):
        seed = args.seed0 + t
        rng = np.random.default_rng(seed)

        node_ids, links, graph = generate_topology(args.n_nodes, seed=seed)
        requests, paths, time_slots, path_resources, valid_paths = generate_requests_and_paths(
            graph, n_requests=args.n_nodes, k_paths=2, n_timeslots=3, seed=seed
        )
        tel = HopFidelityTelemetry(per_hop=0.997, seed=seed)
        active, st = prune(
            requests, paths, time_slots, tel,
            fidelity_threshold=F_MIN, valid_paths=valid_paths, collision_pruning=False,
        )
        conflicts = conflict_pairs_from_paths(active, path_resources)
        qubo = build_qubo(active, requests, conflicts, lambda_margin=1.0)
        n = len(active)
        if n == 0:
            print(f"trial {t:02d} seed={seed}: empty A, skip", flush=True)
            continue

        groups = partition_by_weakest_edge(qubo.Q, N_MAX) if n > N_MAX else [list(range(n))]

        n_match = 0
        energy_gaps = []

        for gi, group in enumerate(groups):
            sub_Q = qubo.Q[np.ix_(group, group)].astype(float)
            sub_Q = 0.5 * (sub_Q + sub_Q.T)
            exact_state, exact_E, n_loc = exact_min_bitstring(sub_Q)

            def fitness(params, sub_Q=sub_Q):
                g = params[:P_DEPTH]
                b = params[P_DEPTH:]
                return run_qaoa(sub_Q, float(np.mean(g)), float(np.mean(b)), p=P_DEPTH).expectation

            gwo_r = gwo_optimize(
                fitness, dim=2 * P_DEPTH,
                n_wolves=args.n_wolves, max_iter=args.max_iter, rng=rng,
            )
            g_b = float(np.mean(gwo_r.best_position[:P_DEPTH]))
            b_b = float(np.mean(gwo_r.best_position[P_DEPTH:]))
            qr = run_qaoa(sub_Q, g_b, b_b, p=P_DEPTH)
            mode_state = int(np.argmax(qr.probabilities))
            mode_E = state_energy(sub_Q, mode_state)
            match = int(mode_state == exact_state or abs(mode_E - exact_E) < 1e-8)

            n_match += match
            energy_gaps.append(mode_E - exact_E)
            rows.append({
                "trial": t, "seed": seed, "partition": gi, "n_part": n_loc,
                "mode_state": mode_state, "exact_state": exact_state, "match": match,
                "mode_E": mode_E, "exact_E": exact_E, "gap_E": mode_E - exact_E,
                "n_active": n, "n_partitions": len(groups),
            })

        rate = n_match / len(groups)
        print(
            f"trial {t:02d} seed={seed} |A|={n} M={len(groups)} "
            f"match={n_match}/{len(groups)} ({100*rate:.1f}%) "
            f"mean_gap_E={float(np.mean(energy_gaps)):.4f} rho={st.reduction_ratio:.3f}",
            flush=True,
        )

    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    matches = [r["match"] for r in rows]
    gaps = np.array([r["gap_E"] for r in rows], dtype=float)
    print("\n=== M3 SUMMARY ===", flush=True)
    print(f"partitions total={len(matches)}", flush=True)
    print(
        f"mode==exact (or equal energy): {sum(matches)}/{len(matches)} = "
        f"{100 * sum(matches) / len(matches):.2f}%",
        flush=True,
    )
    print(f"mean gap_E={gaps.mean():.6f}  median={np.median(gaps):.6f}", flush=True)
    print(f"frac gap_E==0: {100 * np.mean(np.abs(gaps) < 1e-8):.2f}%", flush=True)
    by_size = {}
    for r in rows:
        by_size.setdefault(r["n_part"], []).append(r["match"])
    for s in sorted(by_size):
        m = by_size[s]
        print(f"  n_part={s}: {sum(m)}/{len(m)} = {100 * sum(m) / len(m):.1f}%", flush=True)
    print(f"wrote {out}", flush=True)
    print(f"wall_s={time.time() - t_all:.1f}", flush=True)


if __name__ == "__main__":
    main()
