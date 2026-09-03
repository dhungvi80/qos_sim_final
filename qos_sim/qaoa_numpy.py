"""
qaoa_numpy.py — pure-numpy QAOA statevector simulator.

Stand-in for the Qiskit-backed DVQE kernel (Algorithm 2, Section 4.3.1) for
qubit counts small enough to simulate classically without external
dependencies (this sandbox cannot install qiskit; on the real research
server, replace `run_qaoa` below with genuine Qiskit Aer sub-circuit
execution per Algorithm 2 and keep the same function signature).

For n qubits this does exact statevector simulation: 2^n amplitudes,
practical up to roughly n ~= 14-16 on a laptop, n ~= 20 with more RAM/time.
This matches the *math* Algorithm 2 targets — QAOA is a small unitary
circuit acting on the pruned active subspace — it just skips DVQE's
Hamiltonian-partitioning step (Section 4.3.1), which exists specifically
to keep each sub-circuit within this same small-n regime. For a single
partition (or a small demo scenario), this module and the real DVQE kernel
compute the identical expectation value.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _diag_cost_hamiltonian(Q: np.ndarray) -> np.ndarray:
    """Diagonal of the cost Hamiltonian H_C over all 2^n basis states,
    H_C|x> = (x^T Q x)|x>, so H_C is diagonal in the computational basis.
    Returned as a length-2^n vector indexed by the integer value of the
    bitstring x (bit i = qubit i, i.e. x[i] = (state >> i) & 1).
    """
    n = Q.shape[0]
    dim = 1 << n
    diag = np.zeros(dim)
    # bits[state, i] = i-th bit of `state`
    states = np.arange(dim)[:, None]
    bits = (states >> np.arange(n)) >> 0 & 1  # (dim, n)
    bits = bits.astype(float)
    # x^T Q x for every basis state at once
    diag = np.einsum('si,ij,sj->s', bits, Q, bits)
    return diag


def _apply_mixer(psi: np.ndarray, n: int, beta: float) -> np.ndarray:
    """Apply exp(-i * beta * H_M), H_M = sum_i X_i, to statevector `psi`
    in O(n * 2^n) time by rotating one qubit axis at a time (reshaping the
    statevector into a rank-n tensor), instead of building the full dense
    2^n x 2^n mixer matrix (which costs O(4^n) and is impractical beyond
    n ~= 10).
    """
    c, s = np.cos(beta), -1j * np.sin(beta)
    single = np.array([[c, s], [s, c]], dtype=complex)
    dim = 1 << n
    state = psi.reshape([2] * n)
    for qubit in range(n):
        # move `qubit` axis to front, apply single-qubit gate, move back
        state = np.moveaxis(state, qubit, 0)
        shape_rest = state.shape[1:]
        state = state.reshape(2, -1)
        state = single @ state
        state = state.reshape(2, *shape_rest)
        state = np.moveaxis(state, 0, qubit)
    return state.reshape(dim)


@dataclass
class QAOAResult:
    expectation: float          # <H_C> at the given (gamma, beta)
    probabilities: np.ndarray   # length-2^n measurement distribution
    best_bitstring: int         # argmax probability basis state
    n_qubits: int


def run_qaoa(Q: np.ndarray, gamma: float, beta: float, p: int = 1) -> QAOAResult:
    """Simulate a depth-p QAOA circuit for cost matrix Q and return the
    resulting expectation value and measurement distribution.

    This is the function to swap for real Qiskit Aer execution when
    porting to the research server — same signature, same semantics
    (Algorithm 2 partitions Q first when n is large; call this once per
    sub-Hamiltonian and combine via Eq. (16) as in dvqe_partition.py).
    """
    n = Q.shape[0]
    dim = 1 << n
    diag_HC = _diag_cost_hamiltonian(Q)

    # start in |+>^n (uniform superposition)
    psi = np.ones(dim, dtype=complex) / np.sqrt(dim)

    for _ in range(p):
        # cost unitary: exp(-i*gamma*H_C), diagonal
        psi = psi * np.exp(-1j * gamma * diag_HC)
        # mixer unitary: exp(-i*beta*H_M)
        psi = _apply_mixer(psi, n, beta)

    probs = np.abs(psi) ** 2
    probs = probs / probs.sum()  # renormalize against fp drift
    expectation = float(np.sum(probs * diag_HC))
    best = int(np.argmax(probs))

    return QAOAResult(expectation=expectation, probabilities=probs,
                       best_bitstring=best, n_qubits=n)


def sample_bitstring(result: QAOAResult, rng: np.random.Generator) -> np.ndarray:
    """Draw one measurement outcome from the QAOA output distribution and
    return it as a length-n {0,1} array (bit i = qubit i)."""
    state = rng.choice(1 << result.n_qubits, p=result.probabilities)
    bits = np.array([(state >> i) & 1 for i in range(result.n_qubits)])
    return bits
