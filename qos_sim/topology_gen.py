"""
topology_gen.py — sinh topology + request + path cho kịch bản quy mô lớn
(25/50/100-node), thay thế các scenario nhỏ viết tay trong demo cũ.

Thiết kế:
- Đồ thị: random connected graph (n_nodes, avg_degree~4), độ dài cạnh
  ngẫu nhiên 5-30km — không cố ép khớp số liệu Table 8 gốc của bài báo
  (82/164/328 qubit) vì các số đó tăng tuyến tính "quá đẹp", nghi ngờ là
  ước lượng chứ không phải mô phỏng thật (đã ghi chú trong review trước).
  Số liệu ở đây là REAL, lấy từ mô phỏng thật, dùng để CẬP NHẬT lại
  Table 8 chứ không cố tái tạo số cũ.
- Request: R = n_nodes (1 request/node, giả định tải tỉ lệ quy mô mạng).
- Path: K đường ngắn nhất (k-shortest simple paths, networkx) giữa cặp
  node ngẫu nhiên khác nhau cho mỗi request.
- Time slot: T cố định.
"""

from __future__ import annotations

import itertools
import random

import networkx as nx

from qos_sim.pruning import PathConfig, Request
from qos_sim.simulator import LinkParams


def generate_topology(
    n_nodes: int,
    avg_degree: int = 4,
    length_range: tuple[float, float] = (5.0, 30.0),
    seed: int = 42,
) -> tuple[list[str], dict[tuple[str, str], LinkParams], nx.Graph]:
    """Sinh đồ thị liên thông n_nodes, độ dài cạnh ngẫu nhiên length_range.
    Dùng connected watts-strogatz để có cấu trúc "mạng thực" (không phải
    complete graph, không phải cây thoái hóa)."""
    rng = random.Random(seed)
    k = max(2, min(avg_degree, n_nodes - 1))
    if k % 2 == 1:
        k += 1  # networkx watts_strogatz cần k chẵn

    g = nx.connected_watts_strogatz_graph(n_nodes, k, p=0.2, seed=seed)
    node_ids = [f"N{i}" for i in range(n_nodes)]
    mapping = {i: node_ids[i] for i in range(n_nodes)}
    g = nx.relabel_nodes(g, mapping)

    links: dict[tuple[str, str], LinkParams] = {}
    for a, b in g.edges():
        length = rng.uniform(*length_range)
        links[(a, b)] = LinkParams(length_km=length)
        g[a][b]["weight"] = length  # dùng cho shortest-path theo độ dài thật

    return node_ids, links, g


def generate_requests_and_paths(
    graph: nx.Graph,
    n_requests: int,
    k_paths: int = 2,
    n_timeslots: int = 3,
    weight_range: tuple[float, float] = (1.0, 10.0),
    seed: int = 42,
) -> tuple[list[Request], list[PathConfig], list[int], dict[str, tuple[str, ...]], dict[str, set[str]]]:
    """Sinh n_requests request (cặp nguồn-đích ngẫu nhiên khác nhau),
    mỗi request có k_paths đường (k-shortest simple paths theo độ dài).

    Trả về thêm valid_paths: dict[request_id -> set[path_id]] — chỉ các
    path thực sự nối đúng cặp nguồn-đích của request đó. Truyền map này
    vào prune(valid_paths=...) để tránh xét các cặp (request, path)
    vô nghĩa vật lý (path không nối đúng request).
    """
    rng = random.Random(seed)
    nodes = list(graph.nodes())

    requests: list[Request] = []
    paths: list[PathConfig] = []
    path_resources: dict[str, tuple[str, ...]] = {}
    valid_paths: dict[str, set[str]] = {}
    path_counter = itertools.count()
    pair_to_pathids: dict[tuple[str, str], list[str]] = {}

    for i in range(n_requests):
        src, dst = rng.sample(nodes, 2)
        pair = (src, dst)

        if pair not in pair_to_pathids:
            try:
                gen = nx.shortest_simple_paths(graph, src, dst, weight="weight")
                found = []
                for path_nodes in gen:
                    found.append(path_nodes)
                    if len(found) >= k_paths:
                        break
            except nx.NetworkXNoPath:
                found = []

            path_ids = []
            for path_nodes in found:
                pid = f"path{next(path_counter)}"
                resources = tuple(
                    f"{path_nodes[j]}-{path_nodes[j+1]}"
                    if graph.has_edge(path_nodes[j], path_nodes[j + 1])
                    else f"{path_nodes[j+1]}-{path_nodes[j]}"
                    for j in range(len(path_nodes) - 1)
                )
                paths.append(PathConfig(id=pid, resources=resources))
                path_resources[pid] = resources
                path_ids.append(pid)
            pair_to_pathids[pair] = path_ids

        if not pair_to_pathids[pair]:
            continue  # không có đường nối (hiếm, đồ thị connected nên gần như không xảy ra)

        req_id = f"req{i}"
        requests.append(Request(id=req_id, priority_weight=rng.uniform(*weight_range)))
        valid_paths[req_id] = set(pair_to_pathids[pair])

    time_slots = list(range(n_timeslots))
    return requests, paths, time_slots, path_resources, valid_paths


def generate_hard_conflict_instance(
    n_extra: int = 8,
    seed: int = 42,
    high_weight: float = 10.0,
    med_weight_lo: float = 6.0,
    med_weight_hi: float = 7.5,
) -> tuple[
    list[Request],
    list[Variable],
    list[tuple[int, int]],
    dict[str, float],
]:
    """Hand-crafted conflict structure that forces Greedy into a local optimum.

    Core gadget (classic weighted set-packing trap):
      - Request H with weight high_weight conflicts with both A and B.
      - Requests A, B have weights in [med_weight_lo, med_weight_hi] such
        that w_A + w_B > high_weight but each is < high_weight.
      - Greedy (descending weight) selects H and cannot take A or B.
      - Optimal selects A+B and obtains strictly higher total utility.

    Additional `n_extra` requests share secondary bottleneck resources so
    the active subspace is non-trivial; their weights and pairwise
    conflicts are seed-controlled (seed sweep).

    Returns
    -------
    requests : list[Request]
    variables : list[Variable]  (one variable per request — single path/slot)
    conflicts : list[tuple[int, int]]  (index pairs into variables)
    weights   : dict[request_id, float]  (for convenience)
    """
    from qos_sim.pruning import Variable

    rng = random.Random(seed)

    # --- core gadget ---
    w_h = high_weight
    w_a = rng.uniform(med_weight_lo, med_weight_hi)
    w_b = rng.uniform(med_weight_lo, med_weight_hi)
    # enforce trap condition
    while w_a + w_b <= w_h:
        w_a = rng.uniform(med_weight_lo, med_weight_hi)
        w_b = rng.uniform(med_weight_lo, med_weight_hi)

    req_ids = ["H", "A", "B"]
    weights = {"H": w_h, "A": w_a, "B": w_b}

    # extra requests
    for i in range(n_extra):
        rid = f"E{i}"
        req_ids.append(rid)
        weights[rid] = rng.uniform(1.0, 5.0)

    requests = [Request(id=rid, priority_weight=weights[rid]) for rid in req_ids]
    variables = [
        Variable(request_id=rid, path_id=f"p_{rid}", time_slot=0)
        for rid in req_ids
    ]
    idx = {rid: i for i, rid in enumerate(req_ids)}

    # core conflicts: H--A, H--B  (A and B do NOT conflict)
    conflicts: list[tuple[int, int]] = [
        (idx["H"], idx["A"]),
        (idx["H"], idx["B"]),
    ]

    # secondary bottleneck: a shared resource among a random subset of extras
    # plus occasional cross edges to A/B so the landscape is richer
    extra_ids = [f"E{i}" for i in range(n_extra)]
    # clique on a random subset of extras (dense local conflict)
    clique_size = min(4, n_extra)
    clique = rng.sample(extra_ids, clique_size) if n_extra >= 2 else extra_ids
    for a, b in itertools.combinations(clique, 2):
        conflicts.append((idx[a], idx[b]))

    # random cross edges: each extra conflicts with A or B with p=0.35
    for eid in extra_ids:
        if rng.random() < 0.35:
            target = "A" if rng.random() < 0.5 else "B"
            pair = tuple(sorted((idx[eid], idx[target])))
            if pair not in conflicts and (pair[1], pair[0]) not in conflicts:
                conflicts.append(pair)
        if rng.random() < 0.15:
            pair = tuple(sorted((idx[eid], idx["H"])))
            if pair not in conflicts and (pair[1], pair[0]) not in conflicts:
                conflicts.append(pair)

    # deduplicate
    conflicts = list({(min(a, b), max(a, b)) for a, b in conflicts})
    return requests, variables, conflicts, weights


def hard_instance_to_qubo(
    requests: list[Request],
    variables: list,
    conflicts: list[tuple[int, int]],
):
    """Build a QUBOInstance directly from a hard conflict instance
    (bypasses path/telemetry pruning — combinatorial ablation only)."""
    import numpy as np
    from qos_sim.qubo_builder import QUBOInstance, lambda_star

    n = len(variables)
    weight_of = {r.id: r.priority_weight for r in requests}
    # lambda* from Theorem 1 style bound
    max_w = max(weight_of.values()) if weight_of else 1.0
    lam = max_w + 1.0

    Q = np.zeros((n, n))
    for i, v in enumerate(variables):
        Q[i, i] = -weight_of[v.request_id]
    for a, b in conflicts:
        Q[a, b] += lam
        Q[b, a] += lam

    index_of = {v.key(): i for i, v in enumerate(variables)}
    return QUBOInstance(
        Q=Q,
        index_of=index_of,
        variables=variables,
        lambda_used=lam,
        lambda_star=lam,
    )
