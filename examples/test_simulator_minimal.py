"""
examples/test_simulator_minimal.py — smallest possible smoke test for
simulator.py's build_topology() and run_trace(), which have not been
tested against a real NetSquid installation.

Run this BEFORE trying any larger scenario (25/50/100-node). If it fails,
send the full traceback back for debugging — build_topology()/run_trace()
were written from documentation only and are the most likely place to
need fixes.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qos_sim.pruning import PathConfig
from qos_sim.simulator import LinkParams, build_topology, run_trace


def main():
    print("Step 1: building a minimal 2-node topology...")
    node_ids = ["A", "B"]
    links = {
        ("A", "B"): LinkParams(length_km=10.0),
    }
    network = build_topology(node_ids, links)
    print("  OK: build_topology() returned without error.")
    print(f"  Network object: {network}")

    print("\nStep 2: running a 1-path, 1-timeslot trace...")
    paths = [PathConfig(id="pathAB", resources=("A-B",))]
    time_slots = [0]

    trace = run_trace(network, paths, time_slots, link_params=links)
    print("  OK: run_trace() returned without error.")

    print("\nStep 3: inspecting the result...")
    snapshot = trace[("pathAB", 0)]
    print(f"  fidelity = {snapshot.fidelity:.4f}  "
          f"(expect 0.80-0.98 for a 10km link with depolar_rate=2000 Hz)")
    assert 0.5 < snapshot.fidelity < 1.0, (
        f"Fidelity {snapshot.fidelity} is outside expected range — "
        f"noise is not being applied correctly (see diag_dm_noise2.py)"
    )
    print(f"  available = {snapshot.available}  (expect True — nothing else "
          f"has reserved this resource yet)")

    print("\nAll three steps completed. If the fidelity number looks "
          "physically reasonable to you, build_topology()/run_trace() are "
          "wired correctly for this minimal case — safe to try a slightly "
          "bigger scenario (e.g. 3 nodes, 2 paths) next.")


if __name__ == "__main__":
    main()
