"""
diag_calibrate_rate.py — tìm depolar_rate phù hợp cho link 10km.

NetSquid DepolarNoiseModel: xác suất lỗi mỗi qubit ≈ 1 - exp(-rate * dt_seconds)
Với dt=50000 ns = 5e-5 giây, muốn fidelity ~0.85-0.95 thì cần rate rất nhỏ.

Chạy: python3 examples/diag_calibrate_rate.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import math
from netsquid.components.models import DepolarNoiseModel
from netsquid.qubits import qubitapi as qapi
import netsquid.qubits.ketstates as ks

DELTA_TIME_NS = 50_000  # 10km * 5000 ns/km

print(f"delta_time = {DELTA_TIME_NS} ns  ({DELTA_TIME_NS/1e9:.2e} giây)")
print(f"Lý thuyết: p_error ≈ 1 - exp(-rate * dt_s)")
print()

# Tính lý thuyết
for target_fid in [0.95, 0.90, 0.85, 0.80]:
    # Cho Bell pair: fidelity ≈ (1 + 3*exp(-4/3 * rate * dt_s)) / 4
    # Giải ngược: rate = -ln((4*F-1)/3) * 3/4 / dt_s
    try:
        val = (4 * target_fid - 1) / 3
        if val <= 0:
            continue
        rate_theory = -math.log(val) * 3 / 4 / (DELTA_TIME_NS * 1e-9)
        print(f"Muốn fidelity ≈ {target_fid} -> rate cần ≈ {rate_theory:.2e} Hz")
    except Exception as e:
        print(f"target={target_fid}: {e}")

print()
print("=== Kiểm chứng thực tế ===")

for rate in [10.0, 50.0, 100.0, 500.0, 1000.0, 2000.0, 5000.0]:
    # trung bình 5 lần để bớt nhiễu ngẫu nhiên
    fids = []
    for _ in range(5):
        q1, q2 = qapi.create_qubits(2)
        qapi.assign_qstate([q1, q2], ks.b00)
        m = DepolarNoiseModel(depolar_rate=rate)
        m.error_operation([q2], delta_time=DELTA_TIME_NS)
        fids.append(qapi.fidelity([q1, q2], ks.b00, squared=True))
    avg_fid = sum(fids) / len(fids)
    print(f"rate={rate:>8.1f} Hz  ->  avg fidelity = {avg_fid:.4f}")
