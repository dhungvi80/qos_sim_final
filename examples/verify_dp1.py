"""
verify_dp1.py — S2: kiểm chứng Design Property 1 trên số trial nhỏ.

So sánh:
  U_ILP(A)         — ILP trên active subspace sau unary pruning (collision=False)
  U_ILP(D_forced)  — ILP trên full valid variables, biến bị reject fidelity
                     cố định = 0

Theo Design Property 1 (collision_pruning=False): hai giá trị phải bằng nhau.

Telemetry:
  mặc định  — SyntheticTelemetry (không cần NetSquid)
  --netsquid — NetSquidTelemetry thật (cần netsquid 1.1.x đã cài)

Chạy trên server có NetSquid:
    python3 examples/verify_dp1.py --netsquid --n-nodes 25 --n-trials 15
    python3 examples/verify_dp1.py --netsquid --n-nodes 50 --n-trials 10
    python3 examples/verify_dp1.py --netsquid --n-nodes 100 --n-trials 5

Output: verify_dp1_<n>[_ns].csv trong thư mục gốc repo.
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qos_sim.topology_gen import generate_topology, generate_requests_and_paths
from qos_sim.pruning import prune, NetworkTelemetry, PathConfig, Variable, Request
from qos_sim.qubo_builder import build_qubo, conflict_pairs_from_paths, served_utility
from qos_sim.baseline import exact_solver_highs, greedy_schedule
from qos_sim.repair import verify_feasible

F_MIN = 0.95
EXACT_TIME_LIMIT_S = 30.0


class SyntheticTelemetry(NetworkTelemetry):
    """Fidelity-driven only. Availability always True (matches paper logs).

    Fidelity is deterministic given (seed, path_id, t) so results are
    reproducible. Floor 0.70 + hop-dependent decay produces a realistic
    fraction of rejections at F_MIN=0.95 without needing NetSquid.
    """

    def __init__(self, trial_seed: int):
        self._seed = trial_seed

    def _rng(self, path: PathConfig, t: int):
        import random
        return random.Random(f"{self._seed}|{path.id}|{t}")

    def fidelity(self, path: PathConfig, t: int) -> float:
        rng = self._rng(path, t)
        n_hops = max(1, len(path.resources))
        # ~0.99 per hop base, with small noise → longer paths more likely rejected
        base = 0.99 ** n_hops
        noise = rng.uniform(-0.04, 0.02)
        return max(0.5, min(1.0, base + noise))

    def is_available(self, path: PathConfig, t: int) -> bool:
        return True

    def has_collision(self, path, t, reserved):
        for resource in path.resources:
            holders = reserved.get((resource, t), set())
            if holders and path.id not in holders:
                return True
        return False


def build_full_valid_variables(
    requests: list[Request],
    paths: list[PathConfig],
    time_slots: list[int],
    valid_paths: dict[str, set[str]],
) -> list[Variable]:
    """All topologically valid (i,k,t) — no fidelity filter."""
    paths_by_id = {p.id: p for p in paths}
    variables: list[Variable] = []
    for req in requests:
        pids = sorted(valid_paths.get(req.id, ()))
        for t in time_slots:
            for pid in pids:
                if pid in paths_by_id:
                    variables.append(Variable(req.id, pid, t))
    return variables


def _ilp_with_fixed_zeros(
    variables, requests, conflicts, fixed_zero_idx, time_limit_s
):
    """ILP max sum w_i x_i s.t. conflicts and x_j=0 for j in fixed_zero_idx."""
    import numpy as np
    from scipy.optimize import Bounds, LinearConstraint, milp
    from scipy.sparse import coo_matrix

    n = len(variables)
    weight_of = {r.id: r.priority_weight for r in requests}
    c = -np.array([weight_of[v.request_id] for v in variables], dtype=float)

    lb = np.zeros(n)
    ub = np.ones(n)
    for j in fixed_zero_idx:
        ub[j] = 0.0

    constraints = []
    if conflicts:
        pairs = np.asarray(conflicts, dtype=int)
        rows = np.repeat(np.arange(len(pairs)), 2)
        A = coo_matrix(
            (np.ones(2 * len(pairs)), (rows, pairs.ravel())),
            shape=(len(pairs), n),
        ).tocsr()
        constraints = [LinearConstraint(A, -np.inf, 1.0)]

    t0 = time.perf_counter()
    res = milp(
        c=c,
        integrality=np.ones(n),
        bounds=Bounds(lb, ub),
        constraints=constraints,
        options={"time_limit": float(time_limit_s)},
    )
    runtime = time.perf_counter() - t0
    if res.x is None:
        return float("nan"), False, runtime
    x = np.round(res.x)
    u = sum(
        weight_of[variables[i].request_id] for i in range(n) if x[i] > 0.5
    )
    optimal = bool(res.status == 0)
    return float(u), optimal, runtime


def _make_telemetry(use_netsquid: bool, trial_seed: int,
                    node_ids, links, paths, time_slots):
    if not use_netsquid:
        return SyntheticTelemetry(trial_seed), "synthetic"
    import netsquid as ns
    from netsquid.qubits import QFormalism
    from qos_sim.simulator import NetSquidTelemetry, build_topology, run_trace
    ns.set_qstate_formalism(QFormalism.DM)
    network = build_topology(node_ids, links)
    trace = run_trace(network, paths, time_slots, link_params=links)
    return NetSquidTelemetry(trace), "netsquid"


def run_one(n_nodes: int, trial_seed: int, time_limit_s: float,
            use_netsquid: bool = False) -> dict:
    node_ids, links, graph = generate_topology(n_nodes, seed=trial_seed)
    requests, paths, time_slots, path_resources, valid_paths = generate_requests_and_paths(
        graph, n_requests=n_nodes, k_paths=2, n_timeslots=3, seed=trial_seed
    )
    telemetry, tel_tag = _make_telemetry(
        use_netsquid, trial_seed, node_ids, links, paths, time_slots
    )

    # --- full valid set D (no fidelity filter) ---
    full_vars = build_full_valid_variables(requests, paths, time_slots, valid_paths)
    size_D = len(full_vars)

    # --- unary prune (collision=False) ---
    active_vars, stats = prune(
        requests, paths, time_slots, telemetry,
        fidelity_threshold=F_MIN,
        valid_paths=valid_paths,
        collision_pruning=False,
    )
    size_A = len(active_vars)
    rho = 1.0 - size_A / size_D if size_D else 0.0

    if size_A == 0:
        return {
            "n_nodes": n_nodes, "trial_seed": trial_seed,
            "error": "empty A", "size_D": size_D, "size_A": 0,
        }

    # fidelity mask on full set
    paths_by_id = {p.id: p for p in paths}
    pass_fidelity = []
    for v in full_vars:
        path = paths_by_id[v.path_id]
        fid = telemetry.fidelity(path, v.time_slot)
        avail = telemetry.is_available(path, v.time_slot)
        pass_fidelity.append(fid >= F_MIN and avail)

    n_pass = sum(pass_fidelity)
    assert n_pass == size_A, (
        f"seed={trial_seed}: n_pass={n_pass} != size_A={size_A} "
        f"(rejected_fid={stats.rejected_fidelity})"
    )

    # --- ILP on A (standard pruned reference) ---
    conflicts_A = conflict_pairs_from_paths(active_vars, path_resources)
    qubo_A = build_qubo(active_vars, requests, conflicts_A)
    exact_A = exact_solver_highs(
        qubo_A, requests, conflicts_A, time_limit_s=time_limit_s
    )
    greedy_A = greedy_schedule(qubo_A, requests, conflicts_A)

    # --- ILP on full D with rejected vars fixed to 0 (explicit DP1 check) ---
    conflicts_full = conflict_pairs_from_paths(full_vars, path_resources)
    fixed_zero = [i for i, ok in enumerate(pass_fidelity) if not ok]
    u_forced, opt_forced, rt_forced = _ilp_with_fixed_zeros(
        full_vars, requests, conflicts_full, fixed_zero, time_limit_s
    )

    # DP1: optimum on A must equal optimum on D with unary-infeasible vars fixed 0
    match = (
        exact_A.optimal
        and opt_forced
        and abs(exact_A.served_utility - u_forced) < 1e-5
    )

    return {
        "n_nodes": n_nodes,
        "trial_seed": trial_seed,
        "telemetry": tel_tag,
        "size_D": size_D,
        "size_A": size_A,
        "rho": round(rho, 4),
        "alpha": round(size_A / size_D, 4) if size_D else 0.0,
        "rejected_fidelity": stats.rejected_fidelity,
        "rejected_availability": stats.rejected_availability,
        "rejected_collision": stats.rejected_collision,
        "n_conflicts_A": len(conflicts_A),
        "n_conflicts_full": len(conflicts_full),
        "lambda_star_A": round(qubo_A.lambda_star, 4),
        "exact_A_optimal": exact_A.optimal,
        "U_ILP_A": round(exact_A.served_utility, 6) if exact_A.optimal else float("nan"),
        "U_Greedy_A": round(greedy_A.served_utility, 6),
        "U_ILP_forced": round(u_forced, 6) if opt_forced else float("nan"),
        "forced_optimal": opt_forced,
        "dp1_match": match,
        "exact_A_runtime_s": round(exact_A.runtime_s, 4),
        "forced_runtime_s": round(rt_forced, 4),
    }


def main():
    parser = argparse.ArgumentParser(description="Verify Design Property 1 (S2)")
    parser.add_argument("--n-nodes", type=int, required=True)
    parser.add_argument("--n-trials", type=int, default=15)
    parser.add_argument("--base-seed", type=int, default=42)
    parser.add_argument("--time-limit", type=float, default=EXACT_TIME_LIMIT_S)
    parser.add_argument("--netsquid", action="store_true",
                        help="dùng NetSquidTelemetry thật (cần netsquid đã cài)")
    args = parser.parse_args()

    if args.netsquid:
        try:
            import netsquid  # noqa: F401
        except ImportError:
            print("ERROR: --netsquid yêu cầu package netsquid. "
                  "Cài theo requirements.txt rồi chạy lại.")
            sys.exit(1)

    out_dir = Path(__file__).resolve().parents[1]
    suffix = "_ns" if args.netsquid else ""
    out_path = out_dir / f"verify_dp1_{args.n_nodes}{suffix}.csv"

    rows = []
    tel = "netsquid" if args.netsquid else "synthetic"
    print(f"=== verify_dp1  n_nodes={args.n_nodes}  n_trials={args.n_trials}  "
          f"telemetry={tel} ===")
    for i in range(args.n_trials):
        seed = args.base_seed + i
        t0 = time.perf_counter()
        row = run_one(args.n_nodes, seed, args.time_limit, use_netsquid=args.netsquid)
        elapsed = time.perf_counter() - t0
        rows.append(row)
        status = "OK" if row.get("dp1_match") else ("ERR" if "error" in row else "DIFF")
        print(
            f"  seed={seed:4d}  |D|={row.get('size_D','?'):4}  |A|={row.get('size_A','?'):4}  "
            f"rho={row.get('rho','?'):.3f}  "
            f"rej(f/a/c)={row.get('rejected_fidelity',0)}/"
            f"{row.get('rejected_availability',0)}/"
            f"{row.get('rejected_collision',0)}  "
            f"U_A={row.get('U_ILP_A','nan')}  U_forced={row.get('U_ILP_forced','nan')}  "
            f"[{status}]  {elapsed:.1f}s"
        )

    fieldnames = list(rows[0].keys()) if rows else []
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {out_path}")

    matched = sum(1 for r in rows if r.get("dp1_match"))
    avail0 = sum(1 for r in rows if r.get("rejected_availability", -1) == 0)
    coll0 = sum(1 for r in rows if r.get("rejected_collision", -1) == 0)
    rhos = [r["rho"] for r in rows if "rho" in r]
    print(
        f"\nSummary n={args.n_nodes}: dp1_match={matched}/{len(rows)}  "
        f"availability=0 in {avail0}/{len(rows)}  "
        f"collision=0 in {coll0}/{len(rows)}  "
        f"rho mean={sum(rhos)/len(rhos):.4f}" if rhos else ""
    )


if __name__ == "__main__":
    main()
