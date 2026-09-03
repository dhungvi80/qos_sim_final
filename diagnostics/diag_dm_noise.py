"""
diag_dm_noise.py — kiểm tra cách áp nhiễu đúng trong NetSquid 1.1.8
dùng density matrix representation.

Chạy: python3 examples/diag_dm_noise.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import netsquid as ns
from netsquid.qubits import qubitapi as qapi
import netsquid.qubits.ketstates as ks
import netsquid.qubits.operators as ops

print("=== 1. Kiểm tra representation mặc định ===")
q1, q2 = qapi.create_qubits(2)
print("qrepr:", type(qapi.get_qstate([q1, q2])))
print("qstate:", qapi.get_qstate([q1, q2]))


print("\n=== 2. Tạo Bell pair ở dạng density matrix ===")
q1, q2 = qapi.create_qubits(2, no_state=True)
try:
    from netsquid.qubits import QFormalism, qubitapi as qapi2
    ns.set_qstate_formalism(QFormalism.DM)
    print("Switched to DM formalism OK")
except Exception as e:
    print("DM switch failed:", e)

q1, q2 = qapi.create_qubits(2)
qapi.assign_qstate([q1, q2], ks.b00)
print("fid before:", round(qapi.fidelity([q1, q2], ks.b00, squared=True), 4))


print("\n=== 3. Áp nhiễu qua operators trực tiếp ===")
# NetSquid dùng Kraus operators cho depolarizing channel
# D(rho) = (1-p)*rho + p/3*(X rho X + Y rho Y + Z rho Z)
q1, q2 = qapi.create_qubits(2)
qapi.assign_qstate([q1, q2], ks.b00)
try:
    qapi.operate([q2], ops.X)  # test operate works
    print("operate(X) OK, fid:", round(qapi.fidelity([q1, q2], ks.b00, squared=True), 4))
except Exception as e:
    print("operate failed:", e)


print("\n=== 4. Dùng apply_depolarizing_noise nếu có ===")
q1, q2 = qapi.create_qubits(2)
qapi.assign_qstate([q1, q2], ks.b00)
try:
    qapi.apply_depolarizing_noise(q2, p=0.1)
    print("apply_depolarizing_noise OK, fid:", round(qapi.fidelity([q1, q2], ks.b00, squared=True), 4))
except AttributeError:
    print("apply_depolarizing_noise not found")


print("\n=== 5. Dùng RandomModel / T1T2NoiseModel ===")
q1, q2 = qapi.create_qubits(2)
qapi.assign_qstate([q1, q2], ks.b00)
try:
    from netsquid.components.models.qerrormodels import T1T2NoiseModel
    m = T1T2NoiseModel(T1=1e9, T2=500e6)  # T1=1s, T2=0.5s in nanoseconds
    m.error_operation([q2], delta_time=50000)
    fid = round(qapi.fidelity([q1, q2], ks.b00, squared=True), 4)
    print("T1T2NoiseModel OK, fid:", fid)
except Exception as e:
    print("T1T2NoiseModel failed:", e)


print("\n=== 6. Xem toàn bộ models có sẵn ===")
try:
    import netsquid.components.models.qerrormodels as qem
    print([x for x in dir(qem) if not x.startswith('_')])
except Exception as e:
    print("cannot inspect qerrormodels:", e)
