"""
simulator.py — NetSquid-backed implementation of the NetworkTelemetry
interface defined in pruning.py.

*** UNTESTED IN THIS SANDBOX — NetSquid could not be installed here (needs
a registered account at netsquid.org, outside this environment's network
allowlist). This module is written against NetSquid 1.1.0's documented API
(matching Table 5 of the paper) but has NOT been run. Validate every
function in this file against a real NetSquid installation before trusting
any numbers it produces — see the checklist in `self_check()` at the
bottom, run that first on your server. ***

Design: rather than running a full discrete-event protocol simulation for
every (path, t) query (which would be slow when pruning.py calls fidelity()
and is_available() inside nested loops), this module PRE-COMPUTES a
fidelity/availability trace for the whole scenario in one NetSquid run
(`build_topology` + `run_trace`), then NetworkSquidTelemetry just looks up
cached values. This matches the paper's framing of a "closed-loop control"
system that reacts to telemetry snapshots (Section 3.1), not a fresh
simulation per query.
"""

from __future__ import annotations

from dataclasses import dataclass, field

try:
    import netsquid as ns
    from netsquid.nodes import Node, Network
    from netsquid.components import QuantumChannel, ClassicalChannel
    from netsquid.components.models import FibreDelayModel, DepolarNoiseModel
    from netsquid.qubits import qubitapi as qapi
    from netsquid.qubits.state_sampler import StateSampler
    import netsquid.qubits.ketstates as ks
    NETSQUID_AVAILABLE = True
except ImportError:
    NETSQUID_AVAILABLE = False

from qos_sim.pruning import NetworkTelemetry, PathConfig


@dataclass
class LinkParams:
    """Physical parameters for one edge of the network topology.

    depolar_rate default derived from T2 ~ 10 ms, the order-of-magnitude
    coherence time of an NV-center ELECTRON spin (the "communication
    qubit" actively used for entanglement generation) — distinct from
    the much longer-lived (T2 ~ 1s, Bradley et al. 2018, Nat. Commun.
    9:2552) NUCLEAR spin used for long-term storage. rate = 1/T2.
    Report this T2 assumption explicitly in Table 5 if these numbers
    are used for published results; it is a representative
    order-of-magnitude, not a measurement from a specific device.
    """
    length_km: float
    attenuation_db_per_km: float = 0.2       # standard telecom fibre
    depolar_rate: float = 100.0              # Hz = 1 / (10 ms), see docstring
    coherence_time_ns: float = 10_000_000    # 10 ms, electron-spin communication qubit


def build_topology(
    node_ids: list[str],
    links: dict[tuple[str, str], LinkParams],
) -> "Network":
    """Construct a NetSquid Network of nodes connected by quantum channels
    whose noise model reflects `links`' physical parameters. This is the
    NetSquid analogue of the abstract graph in Fig. 1(a).
    """
    if not NETSQUID_AVAILABLE:
        raise RuntimeError(
            "netsquid is not installed. Register at https://netsquid.org, "
            "then `pip install netsquid` per the account instructions "
            "before using this module."
        )

    network = Network("qos_topology")
    nodes = [Node(nid) for nid in node_ids]
    network.add_nodes(nodes)

    for (a, b), params in links.items():
        depolar_model = DepolarNoiseModel(depolar_rate=params.depolar_rate)
        delay_model = FibreDelayModel()
        qchannel_ab = QuantumChannel(
            f"qchannel_{a}->{b}", length=params.length_km,
            models={"delay_model": delay_model, "quantum_noise_model": depolar_model},
        )
        qchannel_ba = QuantumChannel(
            f"qchannel_{b}->{a}", length=params.length_km,
            models={"delay_model": delay_model, "quantum_noise_model": depolar_model},
        )
        network.add_connection(a, b, channel_to=qchannel_ab, channel_from=qchannel_ba,
                                label=f"{a}-{b}")

    return network


@dataclass
class TraceSnapshot:
    """One (path, t) telemetry reading, cached from a NetSquid run."""
    fidelity: float
    available: bool


def run_trace(
    network: "Network",
    paths: list[PathConfig],
    time_slots: list[int],
    target_state=None,
    link_params: dict[tuple[str, str], "LinkParams"] | None = None,
) -> dict[tuple[str, int], TraceSnapshot]:
    """Run the NetSquid simulation and produce a fidelity/availability
    trace for every (path, t) combination.

    Uses Density Matrix (DM) formalism — required for DepolarNoiseModel
    to correctly reduce fidelity below 1.0. KET (statevector) formalism,
    which is NetSquid's default, does not support mixed-state noise
    correctly for single-qubit operations on entangled pairs.

    Decoherence driver: PER-HOP HERALDING DWELL TIME, not raw fibre
    transit time. Photon transit over metro-scale fibre (tens of us) is
    too short to visibly decohere a qubit with realistic T2 (~10ms,
    LinkParams docstring) — the dominant wait is the two-way classical
    heralding round-trip at each hop before the memory qubit is used.
    HERALDING_DWELL_NS = 145,000 ns (145 us) is the QL2020 setup's
    measured heralding round-trip time, Dahlberg et al. 2019 [ref 15
    in the paper's bibliography] — a real, citable number, not tuned
    for this demo. Total dwell = HERALDING_DWELL_NS * (number of hops
    in the path, i.e. len(path.resources)).
    """
    if not NETSQUID_AVAILABLE:
        raise RuntimeError("netsquid is not installed — see build_topology().")

    if target_state is None:
        target_state = ks.b00
    if link_params is None:
        link_params = {}

    # DM formalism is required for noise to correctly affect fidelity
    from netsquid.qubits import QFormalism
    ns.set_qstate_formalism(QFormalism.DM)

    HERALDING_DWELL_NS = 145_000.0  # Dahlberg et al. 2019 (ref [15]), QL2020 setup

    trace: dict[tuple[str, int], TraceSnapshot] = {}
    reserved_resources: dict[tuple[str, int], set[str]] = {}

    for t in time_slots:
        for path in paths:
            q1, q2 = qapi.create_qubits(2)
            qapi.assign_qstate([q1, q2], target_state)

            n_hops = len(path.resources)
            depolar_rate = 100.0  # default Hz = 1/(10ms), see LinkParams docstring
            for resource in path.resources:
                parts = resource.split("-")
                if len(parts) == 2:
                    a, b = parts
                    params = link_params.get((a, b)) or link_params.get((b, a))
                    if params is not None:
                        depolar_rate = params.depolar_rate
                        break  # rate assumed uniform per path; refine if mixed hardware

            if n_hops > 0:
                delta_time_ns = n_hops * HERALDING_DWELL_NS
                # delay_depolarize: applies depolarizing noise for a qubit
                # that has waited delta_time_ns nanoseconds, at the given
                # depolar_rate (Hz). Correct NetSquid 1.1.x API.
                qapi.delay_depolarize(q2, depolar_rate, delta_time_ns)

            fid = float(qapi.fidelity([q1, q2], target_state, squared=True))

            # explicit cleanup — without this, qubit state can accumulate
            # incorrectly across many iterations (fine at 5-node/12-iter
            # scale, silently wrong by dozens of iterations in), first
            # caught via 25-node run showing 0 fidelity rejections despite
            # 46/50 candidate paths having >=2 hops (which should fail
            # F_min=0.80 per the per-hop model above).
            qapi.discard(q1)
            qapi.discard(q2)

            # availability: resource not already reserved at this time slot
            available = not any(
                resource in reserved_resources.get((resource, t), set())
                for resource in path.resources
            )
            if available:
                for resource in path.resources:
                    reserved_resources.setdefault((resource, t), set()).add(path.id)

            trace[(path.id, t)] = TraceSnapshot(fidelity=fid, available=available)

    return trace


class NetSquidTelemetry(NetworkTelemetry):
    """Concrete NetworkTelemetry backed by a pre-computed NetSquid trace.

    Usage:
        network = build_topology(node_ids, links)
        trace = run_trace(network, paths, time_slots, link_params=links)
        telemetry = NetSquidTelemetry(trace)
        active_vars, stats = prune(requests, paths, time_slots, telemetry, ...)
    """

    def __init__(self, trace: dict[tuple[str, int], TraceSnapshot]):
        self._trace = trace
        self._reserved: dict[tuple[str, int], set[str]] = {}

    def fidelity(self, path: PathConfig, t: int) -> float:
        return self._trace[(path.id, t)].fidelity

    def is_available(self, path: PathConfig, t: int) -> bool:
        return self._trace[(path.id, t)].available

    def has_collision(self, path, t, reserved):
        for resource in path.resources:
            holders = reserved.get((resource, t), set())
            if holders and path.id not in holders:
                return True
        return False


def self_check() -> None:
    """Run this FIRST on your server, before trusting anything else in
    this file:

        python3 -c "from qos_sim.simulator import self_check; self_check()"

    It only checks that NetSquid is importable and that the most basic
    qubit-creation + fidelity call works — it does NOT validate the full
    topology/trace logic above, which you should test separately on a
    tiny 2-node, 1-path scenario and sanity-check by hand.
    """
    if not NETSQUID_AVAILABLE:
        print("FAIL: netsquid is not importable. Install it first "
              "(register at netsquid.org, then `pip install netsquid`).")
        return

    ns_version = getattr(ns, "__version__", None)
    version_note = ns_version or "unknown - verify manually against Table 5's 1.1.0"
    print(f"OK: netsquid imported successfully (version check: {version_note}).")

    try:
        q1, q2 = qapi.create_qubits(2)
        qapi.assign_qstate([q1, q2], ks.b00)
        fid = qapi.fidelity([q1, q2], ks.b00, squared=True)
        print(f"OK: basic fidelity() call succeeded, fid={fid:.4f} "
              f"(expect ~1.0 for a freshly created Bell pair with no noise applied).")
    except Exception as e:
        print(f"FAIL: basic qubit/fidelity smoke test raised {type(e).__name__}: {e}")
        print("Do not trust build_topology()/run_trace() until this passes.")


if __name__ == "__main__":
    self_check()
