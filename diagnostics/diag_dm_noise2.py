"""
diag_dm_noise2.py — tìm đúng cách áp nhiễu trong NetSquid 1.1.8.
Chạy: python3 examples/diag_dm_noise2.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import netsquid as ns
from netsquid.qubits import qubitapi as qapi
import netsquid.qubits.ketstates as ks

print("=== 1. Tất cả hàm trong qapi ===")
all_fns = [x for x in dir(qapi) if not x.startswith('_')]
print(all_fns)

print("\n=== 2. Tất cả models có sẵn ===")
try:
    import netsquid.components.models.qerrormodels as qem
    print([x for x in dir(qem) if not x.startswith('_')])
except Exception as e:
    print("qerrormodels:", e)

print("\n=== 3. Thử T1T2NoiseModel ===")
try:
    from netsquid.components.models.qerrormodels import T1T2NoiseModel
    q1, q2 = qapi.create_qubits(2)
    qapi.assign_qstate([q1, q2], ks.b00)
    print("fid before:", round(qapi.fidelity([q1, q2], ks.b00, squared=True), 4))
    m = T1T2NoiseModel(T1=1e9, T2=500e6)
    m.error_operation([q2], delta_time=50000)
    print("T1T2 fid after:", round(qapi.fidelity([q1, q2], ks.b00, squared=True), 4))
except Exception as e:
    print("T1T2 failed:", e)

print("\n=== 4. Thử đổi sang density matrix formalism ===")
try:
    from netsquid.qubits import QFormalism
    ns.set_qstate_formalism(QFormalism.DM)
    print("DM formalism set OK")
    from netsquid.components.models.qerrormodels import DepolarNoiseModel
    q1, q2 = qapi.create_qubits(2)
    qapi.assign_qstate([q1, q2], ks.b00)
    print("fid before (DM):", round(qapi.fidelity([q1, q2], ks.b00, squared=True), 4))
    m = DepolarNoiseModel(depolar_rate=2000.0)
    m.error_operation([q2], delta_time=50000)
    print("DM depolar fid after:", round(qapi.fidelity([q1, q2], ks.b00, squared=True), 4))
except Exception as e:
    print("DM formalism failed:", e)

print("\n=== 5. Thử T1T2 sau khi chuyển DM ===")
try:
    from netsquid.components.models.qerrormodels import T1T2NoiseModel
    q1, q2 = qapi.create_qubits(2)
    qapi.assign_qstate([q1, q2], ks.b00)
    print("fid before:", round(qapi.fidelity([q1, q2], ks.b00, squared=True), 4))
    m = T1T2NoiseModel(T1=1e9, T2=500e6)
    m.error_operation([q2], delta_time=50000)
    print("T1T2 DM fid after:", round(qapi.fidelity([q1, q2], ks.b00, squared=True), 4))
except Exception as e:
    print("T1T2 DM failed:", e)

print("\n=== 6. Thử Kraus operators thủ công ===")
try:
    ns.set_qstate_formalism(ns.QFormalism.DM)
    import netsquid.qubits.operators as ops
    q1, q2 = qapi.create_qubits(2)
    qapi.assign_qstate([q1, q2], ks.b00)
    p = 0.1  # xác suất lỗi 10%
    qapi.operate([q2], ops.X)
    print("manual X gate fid:", round(qapi.fidelity([q1, q2], ks.b00, squared=True), 4))
except Exception as e:
    print("Kraus failed:", e)
