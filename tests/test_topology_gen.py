import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import networkx as nx
from qos_sim.topology_gen import generate_topology, generate_requests_and_paths


def test_topology_connected_and_sized():
    for n in [10, 25, 50]:
        node_ids, links, g = generate_topology(n, seed=1)
        assert g.number_of_nodes() == n
        assert nx.is_connected(g)
        assert len(node_ids) == n


def test_requests_and_paths_consistent():
    node_ids, links, g = generate_topology(25, seed=1)
    requests, paths, time_slots, path_res, valid_paths = generate_requests_and_paths(
        g, n_requests=25, k_paths=2, n_timeslots=3, seed=1
    )
    assert len(requests) == 25
    assert len(time_slots) == 3
    # every request must have at least one valid path (graph is connected)
    for r in requests:
        assert len(valid_paths[r.id]) >= 1
        assert len(valid_paths[r.id]) <= 2  # k_paths=2


def test_valid_paths_reference_real_edges():
    node_ids, links, g = generate_topology(15, seed=2)
    requests, paths, time_slots, path_res, valid_paths = generate_requests_and_paths(
        g, n_requests=15, k_paths=2, n_timeslots=2, seed=2
    )
    paths_by_id = {p.id: p for p in paths}
    for req in requests:
        for pid in valid_paths[req.id]:
            path = paths_by_id[pid]
            for resource in path.resources:
                a, b = resource.split("-")
                assert g.has_edge(a, b), f"path {pid} uses non-existent edge {resource}"


def test_d_size_matches_valid_paths_sum():
    node_ids, links, g = generate_topology(20, seed=3)
    requests, paths, time_slots, path_res, valid_paths = generate_requests_and_paths(
        g, n_requests=20, k_paths=2, n_timeslots=3, seed=3
    )
    expected_D = sum(len(valid_paths[r.id]) for r in requests) * len(time_slots)
    assert expected_D > 0
    # sanity: should be well under the naive (and wrong) full cross-product
    naive_D = len(requests) * len(paths) * len(time_slots)
    assert expected_D <= naive_D


if __name__ == "__main__":
    test_topology_connected_and_sized()
    test_requests_and_paths_consistent()
    test_valid_paths_reference_real_edges()
    test_d_size_matches_valid_paths_sum()
    print("All tests passed.")
