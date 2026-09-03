"""
diag_noise_api.py — tìm đúng API áp nhiễu trong NetSquid 1.1.8.

Chạy: python3 examples/diag_noise_api.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from netsquid.components.models import DepolarNoiseModel
from netsquid.qubits import qubitapi as qapi
import netsquid.qubits.ketstates as ks
import inspect


print("=== 1. error_operation signature ===")
m = DepolarNoiseModel(depolar_rate=1e6)
try:
    print(inspect.signature(m.error_operation))
except Exception as e:
    print("cannot inspect:", e)


print("\n=== 2. qapi depolarize / noise functions ===")
noise_fns = [x for x in dir(qapi) if any(
    k in x.lower() for k in ['depol', 'noise', 'error', 'channel', 'damp', 'apply']
)]
print(noise_fns)


print("\n=== 3. try qapi.depolarize ===")
q1, q2 = qapi.create_qubits(2)
qapi.assign_qstate([q1, q2], ks.b00)
print("before:", round(qapi.fidelity([q1, q2], ks.b00, squared=True), 4))
try:
    qapi.depolarize(q2, prob=0.15)
    print("after depolarize(prob=0.15):",
          round(qapi.fidelity([q1, q2], ks.b00, squared=True), 4))
except Exception as e:
    print("depolarize failed:", e)


print("\n=== 4. try error_operation with different arg names ===")
for kwargs in [
    {"delta_time": 50000},
    {"time": 50000},
    {"t": 50000},
    {},
]:
    q1, q2 = qapi.create_qubits(2)
    qapi.assign_qstate([q1, q2], ks.b00)
    m2 = DepolarNoiseModel(depolar_rate=1e6)
    try:
        m2.error_operation([q2], **kwargs)
        fid = round(qapi.fidelity([q1, q2], ks.b00, squared=True), 4)
        print(f"  error_operation([q2], **{kwargs}) -> fidelity={fid}")
    except Exception as e:
        print(f"  error_operation([q2], **{kwargs}) -> FAILED: {e}")


print("\n=== 5. check compute_model ===")
q1, q2 = qapi.create_qubits(2)
qapi.assign_qstate([q1, q2], ks.b00)
m3 = DepolarNoiseModel(depolar_rate=1e6)
try:
    result = m3.compute_model([q2], delta_time=50000)
    print("compute_model result:", result)
    fid = round(qapi.fidelity([q1, q2], ks.b00, squared=True), 4)
    print("fidelity after compute_model:", fid)
except Exception as e:
    print("compute_model failed:", e)
