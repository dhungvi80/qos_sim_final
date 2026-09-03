# qos_sim — Mô phỏng thực nghiệm cho bài báo Quantum OS Resource Allocation

Codebase mô phỏng đầy đủ cho bài báo *"A Quantum Optimization Approach for
Dynamic Resource Allocation in Quantum Networks"*. Triển khai Algorithm 1-4,
Proposition 1-2, Theorem 1 như mô tả trong bản thảo, chạy telemetry thật qua
NetSquid 1.1.8 trên server.

Xem `SUMMARY.md` để biết toàn bộ quá trình xây dựng, các bug đã phát hiện/sửa,
và phát hiện khoa học chính (QAOA+Repair hội tụ về Greedy Scheduler — xem
Proposition 2 và Mục 7.7 của bản thảo).

## Cấu trúc

```
qos_sim/
├── qos_sim/                   # package chính
│   ├── pruning.py              # Algorithm 1 — kiểm định Proposition 1
│   ├── qubo_builder.py         # QUBO + λ* (Theorem 1)
│   ├── qaoa_numpy.py           # QAOA statevector (numpy, thay Qiskit)
│   ├── gwo_optimizer.py        # Grey Wolf Optimizer
│   ├── dvqe_partition.py       # Algorithm 2 — weakest-edge cut + Eq.16
│   ├── repair.py               # Algorithm 3 mở rộng — kiểm định Proposition 2
│   ├── simulator.py            # NetSquid telemetry thật
│   ├── topology_gen.py         # Sinh topology/request/path quy mô 25-100 node
│   └── baseline.py             # Greedy, SA, Tabu, GA, ACO
├── tests/                      # 29 test đơn vị, tất cả pass
├── examples/                   # Demo + script chạy thực nghiệm chính
├── diagnostics/                # Script chẩn đoán dùng khi debug NetSquid trên server
└── SUMMARY.md                  # Tổng kết toàn bộ quá trình
```

## Cài đặt

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

NetSquid cần đăng ký tài khoản riêng tại [netsquid.org](https://netsquid.org)
trước khi cài — xem hướng dẫn trong `requirements.txt`. Các module toán học
(`pruning.py`, `qubo_builder.py`, `dvqe_partition.py`, `repair.py`,
`baseline.py`, `topology_gen.py`) **không cần NetSquid**, chạy được ngay.



## Import path

From the repository root:

```bash
export PYTHONPATH=.
python3 tests/test_pruning.py
python3 examples/demo_end_to_end.py
```

## Chạy test

```bash
python3 tests/test_pruning.py
python3 tests/test_qubo_and_repair.py
python3 tests/test_dvqe_partition.py
python3 tests/test_baseline.py
python3 tests/test_topology_gen.py
```

## Chạy thực nghiệm

**Không cần NetSquid** (dữ liệu giả lập, kiểm tra logic):
```bash
python3 examples/demo_end_to_end.py
python3 examples/demo_pruning.py
```

**Cần NetSquid đã cài + đăng nhập server** (dữ liệu thật, dùng cho Table 6-8
của bài báo):
```bash
python3 examples/test_simulator_minimal.py     # kiểm tra NetSquid hoạt động trước
python3 examples/demo_netsquid_pipeline.py      # demo 5-node, telemetry thật
python3 examples/run_100_trials.py --n-nodes 25 # thực nghiệm đầy đủ, 100 trial
python3 examples/run_100_trials.py --n-nodes 50
python3 examples/run_100_trials.py --n-nodes 100
```

Nếu đã có `results_raw_*.csv` từ lần chạy trước nhưng thiếu cột (ví dụ mới
thêm thuật toán baseline), tính lại tổng hợp mà **không cần chạy lại**:
```bash
python3 examples/recompute_summary.py
```

## Kết quả chính (100 trial × 3 quy mô, xem SUMMARY.md mục 5)

| n_nodes | Reduction Ratio | QAOA+Repair U(x) | Feasibility |
|---|---|---|---|
| 25  | 34.4% ± 5.6% | 84.18 ± 13.57 | **100%** |
| 50  | 25.1% ± 3.2% | 131.73 ± 16.85 | **100%** |
| 100 | 18.3% ± 2.4% | 199.08 ± 26.12 | **100%** |

QAOA+Repair đạt utility **giống hệt Greedy Scheduler** ở mọi trial (Proposition
2 chứng minh + xác nhận thực nghiệm). Baseline không đảm bảo feasibility
(Simulated Annealing, Tabu, GA, ACO) có utility thô cao hơn nhưng feasibility
sụp đổ theo quy mô — Ant Colony Optimization: 26% (25-node) → 0% (100-node).

## Giới hạn đã biết

- `qaoa_numpy.py` thay thế Qiskit thật (sandbox phát triển không cài được
  Qiskit) — cùng toán học ở quy mô nhỏ (≤10-14 qubit/partition), nhưng chưa
  chạy trên Qiskit Aer/phần cứng lượng tử thật
- Chưa có so sánh với Gurobi (exact solver) — cần license WLS hoạt động
- Xem `SUMMARY.md` mục 7 để biết đầy đủ các việc còn tồn đọng

## License

MIT — xem `LICENSE`.
