"""
gwo_optimizer.py — Grey Wolf Optimizer for QAOA parameter search
(Section 4.3.2, paper reference [14]: Mirjalili, Mirjalili & Lewis, 2014).

Standard GWO: a population ("pack") of candidate solutions ("wolves") is
guided by the three best-so-far wolves (alpha, beta, delta) toward better
regions of the search space. Here each wolf is a QAOA parameter vector
(gamma_1..p, beta_1..p) for depth-p QAOA, and fitness = -<H_C> (since GWO
minimizes by convention and we want to MINIMIZE the QUBO objective, i.e.
maximize -<H_C> is wrong — we directly minimize <H_C>, see `fitness_fn`
usage in orchestrator.py).

Per Section 4.6 (Convergence Properties) and Section 4.3.2 (Table 4a):
this optimizer has NO global-optimum guarantee on non-convex landscapes,
and does not resolve barren plateaus (Arrasmith et al. [13]) at scale.
Treat its output as a local/heuristic search result, not a certified
optimum — the paper's feasibility guarantee comes from Constraint Repair
(Algorithm 3), not from this optimizer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass
class GWOResult:
    best_position: np.ndarray   # best (gamma, beta, ...) found
    best_fitness: float         # fitness at best_position (lower = better)
    history: list[float]        # best_fitness per iteration, for convergence plots


def optimize(
    fitness_fn: Callable[[np.ndarray], float],
    dim: int,
    bounds: tuple[float, float] = (0.0, 2 * np.pi),
    n_wolves: int = 20,
    max_iter: int = 50,
    rng: np.random.Generator | None = None,
) -> GWOResult:
    """Minimize `fitness_fn` over a `dim`-dimensional box `bounds` using GWO.

    dim should be 2*p for depth-p QAOA (p gamma values + p beta values).
    Default max_iter=50 matches the paper's stated empirical convergence
    window (Section 4.3.2 / 4.3 text: "GWO converges within 50 iterations").
    """
    rng = rng or np.random.default_rng()
    lo, hi = bounds

    positions = rng.uniform(lo, hi, size=(n_wolves, dim))
    fitness = np.array([fitness_fn(pos) for pos in positions])

    order = np.argsort(fitness)
    alpha_pos, beta_pos, delta_pos = positions[order[:3]]
    alpha_fit = fitness[order[0]]

    history = [alpha_fit]

    for t in range(max_iter):
        a = 2.0 - 2.0 * t / max(max_iter - 1, 1)  # linearly decreases 2 -> 0

        for i in range(n_wolves):
            new_pos = np.zeros(dim)
            for leader_pos in (alpha_pos, beta_pos, delta_pos):
                r1, r2 = rng.random(dim), rng.random(dim)
                A = 2 * a * r1 - a
                C = 2 * r2
                D = np.abs(C * leader_pos - positions[i])
                new_pos += leader_pos - A * D
            positions[i] = np.clip(new_pos / 3.0, lo, hi)

        fitness = np.array([fitness_fn(pos) for pos in positions])
        order = np.argsort(fitness)
        if fitness[order[0]] < alpha_fit:
            alpha_fit = fitness[order[0]]
        alpha_pos, beta_pos, delta_pos = positions[order[:3]]
        history.append(alpha_fit)

    return GWOResult(best_position=alpha_pos, best_fitness=alpha_fit, history=history)
