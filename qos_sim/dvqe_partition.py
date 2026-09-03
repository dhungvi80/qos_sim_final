"""
dvqe_partition.py — Algorithm 2 partitioning (Section 4.3.1): cut the QUBO
coupling graph along its WEAKEST edges, keep strong couplings intra-partition.

Method: greedy union-find, strongest edges merged first, capped at
max_size per partition. Edges never merged (because they'd exceed
max_size) are exactly the weakest links left uncut by the merge order —
this is the "cut along weakest coupling terms" rule from the paper.
"""

from __future__ import annotations

import numpy as np


class _UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
        self.size = [1] * n

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return True
        self.parent[ra] = rb
        self.size[rb] += self.size[ra]
        return True


def partition_by_weakest_edge(Q: np.ndarray, max_size: int) -> list[list[int]]:
    """Return a list of qubit-index groups (partitions), each of size
    <= max_size, built by greedily merging along the STRONGEST |Q[i,j]|
    edges first — so any edge left uncut is necessarily one of the
    remaining weakest couplings in the graph."""
    n = Q.shape[0]
    edges = [(abs(Q[i, j]), i, j) for i in range(n) for j in range(i + 1, n)
              if Q[i, j] != 0]
    edges.sort(reverse=True)  # strongest first

    uf = _UnionFind(n)
    for _, i, j in edges:
        ri, rj = uf.find(i), uf.find(j)
        if ri != rj and uf.size[ri] + uf.size[rj] <= max_size:
            uf.union(i, j)

    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(uf.find(i), []).append(i)
    return list(groups.values())


def reconstruct_expectation(
    Q: np.ndarray,
    partitions: list[list[int]],
    partition_marginals: list[np.ndarray],
) -> float:
    """Eq. (16): <H_C> = sum_m <H_C^(m)> + sum_boundary Q_ij <x_i x_j>.

    partition_marginals[m][k] = P(x_k = 1) for the k-th qubit in
    partitions[m], from that partition's QAOA output. Boundary terms
    (i, j in different partitions) use the independence approximation
    <x_i x_j> ~= <x_i><x_j>, since boundary qubits were optimized in
    separate circuits with no joint measurement available.
    """
    marg = np.zeros(Q.shape[0])
    for part, m in zip(partitions, partition_marginals):
        for local_idx, global_idx in enumerate(part):
            marg[global_idx] = m[local_idx]

    owner = {}
    for pid, part in enumerate(partitions):
        for idx in part:
            owner[idx] = pid

    total = 0.0
    n = Q.shape[0]
    for i in range(n):
        total += Q[i, i] * marg[i]
        for j in range(i + 1, n):
            if Q[i, j] == 0:
                continue
            if owner[i] == owner[j]:
                continue  # already counted inside that partition's <H_C^(m)>
            total += 2 * Q[i, j] * marg[i] * marg[j]  # boundary term, Eq. 16
    return total


def marginals_from_probabilities(probs: np.ndarray, n_qubits: int) -> np.ndarray:
    """P(x_k=1) for each qubit k, from a QAOA output distribution over
    2^n_qubits basis states (qaoa_numpy.QAOAResult.probabilities)."""
    marg = np.zeros(n_qubits)
    for state, p in enumerate(probs):
        for k in range(n_qubits):
            if (state >> k) & 1:
                marg[k] += p
    return marg
