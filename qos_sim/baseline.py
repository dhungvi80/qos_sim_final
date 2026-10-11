"""
baseline.py — Greedy và Simulated Annealing, đối chứng cho Table 6.

Cả 2 baseline dùng chung QUBOInstance/conflicts với pipeline chính
(qubo_builder.py) để so sánh công bằng trên cùng active subspace A.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

import numpy as np

from qos_sim.pruning import Request, Variable
from qos_sim.qubo_builder import QUBOInstance, served_utility
from qos_sim.repair import repair, verify_feasible


@dataclass
class BaselineResult:
    x: np.ndarray
    feasible: bool
    served_utility: float
    runtime_s: float


def random_init_and_repair(
    qubo: QUBOInstance,
    requests: list[Request],
    conflicts: list[tuple[int, int]],
    seed: int | None = None,
    p_activate: float = 0.5,
) -> BaselineResult:
    """Ablation baseline: random bitstring on the active subspace, then
    Algorithm 3 (repair + completion). Isolates the contribution of the
    classical repair layer from any variational search (QAOA/GWO).

    p_activate controls the Bernoulli probability of setting each variable
    to 1 before repair. Default 0.5 matches an uninformed random start.
    """
    import time
    t0 = time.perf_counter()
    rng = random.Random(seed)

    n = len(qubo.variables)
    x_raw = np.array([1.0 if rng.random() < p_activate else 0.0 for _ in range(n)])
    x = repair(qubo, x_raw, requests, conflicts)
    runtime = time.perf_counter() - t0
    return BaselineResult(
        x=x,
        feasible=verify_feasible(x, conflicts),
        served_utility=served_utility(qubo, x),
        runtime_s=runtime,
    )


def exact_solver_pulp(
    qubo: QUBOInstance,
    requests: list[Request],
    conflicts: list[tuple[int, int]],
    time_limit_s: float = 60.0,
) -> BaselineResult:
    """Exact ILP solver via PuLP + CBC (open-source).

    Maximises served utility subject to pairwise exclusivity constraints.
    Equivalent to the constrained form before QUBO penalisation, so the
    optimum is a valid reference for Optimality Gap reporting.
    """
    import time
    import pulp

    t0 = time.perf_counter()
    n = len(qubo.variables)
    weight_of = {r.id: r.priority_weight for r in requests}

    prob = pulp.LpProblem("quantum_resource_allocation", pulp.LpMaximize)
    x_vars = [pulp.LpVariable(f"x_{i}", cat="Binary") for i in range(n)]

    prob += pulp.lpSum(
        weight_of[qubo.variables[i].request_id] * x_vars[i] for i in range(n)
    )

    for a, b in conflicts:
        prob += x_vars[a] + x_vars[b] <= 1

    solver = pulp.PULP_CBC_CMD(msg=False, timeLimit=time_limit_s)
    status = prob.solve(solver)

    x = np.zeros(n)
    if status == pulp.LpStatusOptimal:
        for i in range(n):
            val = pulp.value(x_vars[i])
            x[i] = 1.0 if val is not None and val > 0.5 else 0.0

    runtime = time.perf_counter() - t0
    feas = verify_feasible(x, conflicts) if status == pulp.LpStatusOptimal else False
    return BaselineResult(
        x=x,
        feasible=feas,
        served_utility=served_utility(qubo, x) if feas else 0.0,
        runtime_s=runtime,
    )


@dataclass
class ExactResult:
    x: np.ndarray
    served_utility: float      # NaN if no incumbent was found
    runtime_s: float
    optimal: bool              # True only if the solver proved optimality
    # Valid UPPER bound on the optimum (HiGHS dual bound). Equals served_utility
    # when optimal; NaN if the solver reported none. Lets callers report a valid
    # optimality-gap bound (U_bound - U) / U_bound even when the time limit hits.
    bound: float = float("nan")


def exact_solver_highs(
    qubo: QUBOInstance,
    requests: list[Request],
    conflicts: list[tuple[int, int]],
    time_limit_s: float = 60.0,
) -> ExactResult:
    """Exact reference optimum of the constrained problem
        max sum_i w_i x_i   s.t.  x_a + x_b <= 1  for every conflict (a, b),
    solved with HiGHS through scipy.optimize.milp (no CBC binary needed).

    Same ILP as exact_solver_pulp. `optimal` is False when the time limit is
    hit or the solver fails; callers must then NOT report an optimality gap
    (the old PuLP path silently produced gap 0 in that case).
    """
    import time
    from scipy.optimize import Bounds, LinearConstraint, milp
    from scipy.sparse import coo_matrix

    t0 = time.perf_counter()
    n = len(qubo.variables)
    if n == 0:
        return ExactResult(np.zeros(0), 0.0, time.perf_counter() - t0, True, 0.0)

    weight_of = {r.id: r.priority_weight for r in requests}
    c = -np.array([weight_of[v.request_id] for v in qubo.variables], dtype=float)

    constraints = []
    if conflicts:
        pairs = np.asarray(conflicts, dtype=int)
        rows = np.repeat(np.arange(len(pairs)), 2)
        A = coo_matrix((np.ones(2 * len(pairs)), (rows, pairs.ravel())),
                       shape=(len(pairs), n)).tocsr()
        constraints = [LinearConstraint(A, -np.inf, 1.0)]

    res = milp(c=c, integrality=np.ones(n), bounds=Bounds(0, 1),
               constraints=constraints, options={"time_limit": float(time_limit_s)})
    runtime = time.perf_counter() - t0

    # milp minimises -U, so the dual (lower) bound on -U is -upper bound on U.
    dual = getattr(res, "mip_dual_bound", None)
    bound = -float(dual) if dual is not None and np.isfinite(dual) else float("nan")

    if res.x is None:
        return ExactResult(np.zeros(n), float("nan"), runtime, False, bound)
    x = np.round(res.x)
    feasible = verify_feasible(x, conflicts)
    u = served_utility(qubo, x)
    optimal = bool(res.status == 0 and feasible)
    if optimal:
        bound = u          # proven optimum: bound is tight
    elif np.isfinite(bound):
        bound = max(bound, u)   # an upper bound can never be below an incumbent
    return ExactResult(x=x, served_utility=u, runtime_s=runtime,
                       optimal=optimal, bound=bound)


def greedy_schedule(
    qubo: QUBOInstance,
    requests: list[Request],
    conflicts: list[tuple[int, int]],
) -> BaselineResult:

    """Greedy: xét biến theo thứ tự trọng số giảm dần, nhận nếu không
    xung đột với tập đã chọn. Feasible-by-construction (không cần repair).
    """
    import time
    t0 = time.perf_counter()

    weight_of = {r.id: r.priority_weight for r in requests}
    n = len(qubo.variables)
    order = sorted(range(n), key=lambda i: -weight_of[qubo.variables[i].request_id])

    conflict_set: dict[int, set[int]] = {i: set() for i in range(n)}
    for a, b in conflicts:
        conflict_set[a].add(b)
        conflict_set[b].add(a)

    x = np.zeros(n)
    selected: set[int] = set()
    for i in order:
        if not (conflict_set[i] & selected):
            x[i] = 1.0
            selected.add(i)

    runtime = time.perf_counter() - t0
    return BaselineResult(
        x=x, feasible=True,  # đúng theo cấu trúc, luôn feasible
        served_utility=served_utility(qubo, x), runtime_s=runtime,
    )


def simulated_annealing(
    qubo: QUBOInstance,
    conflicts: list[tuple[int, int]],
    n_iters: int = 2000,
    t_start: float = 10.0,
    t_end: float = 0.01,
    seed: int | None = None,
) -> BaselineResult:
    """Simulated Annealing tối ưu trực tiếp trên hàm mục tiêu QUBO
    (x^T Q x, Eq. 13) — bao gồm cả penalty term, nên KHÔNG đảm bảo
    feasible tuyệt đối (khác Greedy) — đúng đặc trưng của SA cổ điển,
    dùng để đối chứng công bằng với "Raw QAOA" trước khi qua Algorithm 3.
    """
    import time
    t0 = time.perf_counter()
    rng = random.Random(seed)

    n = len(qubo.variables)
    Q = qubo.Q
    x = np.array([rng.randint(0, 1) for _ in range(n)], dtype=float)

    def energy(vec):
        return float(vec @ Q @ vec)

    current_e = energy(x)
    best_x, best_e = x.copy(), current_e

    for it in range(n_iters):
        frac = it / max(n_iters - 1, 1)
        temp = t_start * (t_end / t_start) ** frac  # geometric cooling

        i = rng.randrange(n)
        x_new = x.copy()
        x_new[i] = 1.0 - x_new[i]
        new_e = energy(x_new)

        delta = new_e - current_e
        if delta < 0 or rng.random() < math.exp(-delta / max(temp, 1e-9)):
            x, current_e = x_new, new_e
            if current_e < best_e:
                best_x, best_e = x.copy(), current_e

    feasible = all(not (best_x[a] > 0.5 and best_x[b] > 0.5) for a, b in conflicts)
    runtime = time.perf_counter() - t0
    return BaselineResult(
        x=best_x, feasible=feasible,
        served_utility=served_utility(qubo, best_x), runtime_s=runtime,
    )


def tabu_search(
    qubo: QUBOInstance,
    conflicts: list[tuple[int, int]],
    n_iters: int = 2000,
    tabu_tenure: int = 15,
    seed: int | None = None,
) -> BaselineResult:
    """Tabu Search: local search bit-flip, cấm quay lại chỉ số vừa lật
    trong `tabu_tenure` vòng gần nhất (trừ khi đạt best-so-far — aspiration
    criterion chuẩn). Cùng năng lượng QUBO như SA nhưng tìm kiếm có định
    hướng hơn (luôn chọn hàng xóm tốt nhất khả dụng, không random-walk).
    """
    import time
    t0 = time.perf_counter()
    rng = random.Random(seed)

    n = len(qubo.variables)
    Q = qubo.Q

    def energy(vec):
        return float(vec @ Q @ vec)

    x = np.array([rng.randint(0, 1) for _ in range(n)], dtype=float)
    current_e = energy(x)
    best_x, best_e = x.copy(), current_e

    tabu_until = np.zeros(n, dtype=int)  # index i cấm lật đến vòng tabu_until[i]

    for it in range(n_iters):
        best_move, best_move_e = None, math.inf
        for i in range(n):
            x_try = x.copy()
            x_try[i] = 1.0 - x_try[i]
            e = energy(x_try)
            is_tabu = tabu_until[i] > it
            aspiration = e < best_e  # aspiration criterion: cho phép phá tabu nếu tốt hơn best-so-far
            if (not is_tabu or aspiration) and e < best_move_e:
                best_move, best_move_e = i, e

        if best_move is None:
            break
        x[best_move] = 1.0 - x[best_move]
        current_e = best_move_e
        tabu_until[best_move] = it + tabu_tenure

        if current_e < best_e:
            best_x, best_e = x.copy(), current_e

    feasible = all(not (best_x[a] > 0.5 and best_x[b] > 0.5) for a, b in conflicts)
    runtime = time.perf_counter() - t0
    return BaselineResult(
        x=best_x, feasible=feasible,
        served_utility=served_utility(qubo, best_x), runtime_s=runtime,
    )


def genetic_algorithm(
    qubo: QUBOInstance,
    conflicts: list[tuple[int, int]],
    population_size: int = 40,
    n_generations: int = 100,
    mutation_rate: float = 0.02,
    crossover_rate: float = 0.8,
    elitism: int = 2,
    seed: int | None = None,
) -> BaselineResult:
    """Genetic Algorithm: quần thể bitstring, tournament selection,
    uniform crossover, bit-flip mutation, elitism giữ lại `elitism` cá
    thể tốt nhất mỗi thế hệ không qua lai/đột biến.
    """
    import time
    t0 = time.perf_counter()
    rng = random.Random(seed)

    n = len(qubo.variables)
    Q = qubo.Q

    def energy(vec):
        return float(vec @ Q @ vec)

    population = [np.array([rng.randint(0, 1) for _ in range(n)], dtype=float)
                  for _ in range(population_size)]

    def tournament_select(pop, fits, k=3):
        idxs = rng.sample(range(len(pop)), k)
        best = min(idxs, key=lambda i: fits[i])
        return pop[best]

    best_x, best_e = None, math.inf

    for gen in range(n_generations):
        fits = [energy(ind) for ind in population]
        gen_best_idx = int(np.argmin(fits))
        if fits[gen_best_idx] < best_e:
            best_e = fits[gen_best_idx]
            best_x = population[gen_best_idx].copy()

        order = sorted(range(population_size), key=lambda i: fits[i])
        new_pop = [population[i].copy() for i in order[:elitism]]

        while len(new_pop) < population_size:
            p1 = tournament_select(population, fits)
            p2 = tournament_select(population, fits)
            if rng.random() < crossover_rate:
                mask = np.array([rng.random() < 0.5 for _ in range(n)])
                child = np.where(mask, p1, p2)
            else:
                child = p1.copy()
            for i in range(n):
                if rng.random() < mutation_rate:
                    child[i] = 1.0 - child[i]
            new_pop.append(child)

        population = new_pop[:population_size]

    feasible = all(not (best_x[a] > 0.5 and best_x[b] > 0.5) for a, b in conflicts)
    runtime = time.perf_counter() - t0
    return BaselineResult(
        x=best_x, feasible=feasible,
        served_utility=served_utility(qubo, best_x), runtime_s=runtime,
    )


def ant_colony_optimization(
    qubo: QUBOInstance,
    conflicts: list[tuple[int, int]],
    n_ants: int = 30,
    n_iterations: int = 60,
    evaporation: float = 0.1,
    alpha: float = 1.0,
    seed: int | None = None,
) -> BaselineResult:
    """Ant Colony Optimization cho bài toán QUBO nhị phân (Binary Ant
    System): mỗi biến x_i có pheromone tau_i biểu thị xu hướng đặt x_i=1.
    Mỗi ant xây solution bằng cách lấy mẫu Bernoulli(p_i) với
    p_i = tau_i^alpha / (tau_i^alpha + (1-tau_i)^alpha). Sau mỗi vòng,
    pheromone bay hơi rồi được củng cố theo chất lượng nghiệm tốt nhất
    vòng đó (giống MMAS - Max-Min Ant System, đơn giản hóa).
    """
    import time
    t0 = time.perf_counter()
    rng = random.Random(seed)

    n = len(qubo.variables)
    Q = qubo.Q

    def energy(vec):
        return float(vec @ Q @ vec)

    tau = np.full(n, 0.5)  # pheromone khởi tạo trung lập
    tau_min, tau_max = 0.05, 0.95

    best_x, best_e = None, math.inf

    for iteration in range(n_iterations):
        ants = []
        for _ in range(n_ants):
            p = tau ** alpha / (tau ** alpha + (1 - tau) ** alpha)
            x = np.array([1.0 if rng.random() < p[i] else 0.0 for i in range(n)])
            ants.append((x, energy(x)))

        ants.sort(key=lambda t: t[1])
        iter_best_x, iter_best_e = ants[0]
        if iter_best_e < best_e:
            best_e = iter_best_e
            best_x = iter_best_x.copy()

        # bay hơi + củng cố theo nghiệm tốt nhất của vòng này
        tau = (1 - evaporation) * tau + evaporation * iter_best_x
        tau = np.clip(tau, tau_min, tau_max)

    feasible = all(not (best_x[a] > 0.5 and best_x[b] > 0.5) for a, b in conflicts)
    runtime = time.perf_counter() - t0
    return BaselineResult(
        x=best_x, feasible=feasible,
        served_utility=served_utility(qubo, best_x), runtime_s=runtime,
    )
