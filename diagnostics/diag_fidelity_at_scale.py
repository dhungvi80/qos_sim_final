"""
diag_fidelity_at_scale.py — kiểm tra fidelity có đúng giảm theo số hop
khi chạy nhiều path liên tiếp (mô phỏng tình huống 25-node, 50 path).

Chạy: python3 examples/diag_fidelity_at_scale.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import netsquid as ns
from netsquid.qubits import qubitapi as qapi, QFormalism
import netsquid.qubits.ketstates as ks

ns.set_qstate_formalism(QFormalism.DM)

HERALDING_DWELL_NS = 145_000.0
depolar_rate = 100.0

print("Mô phỏng 50 path liên tiếp, hop count ngẫu nhiên 1-7 (giống 25-node)...")
print(f"{'iter':>4} {'hops':>4} {'fidelity':>10}")

import random
rng = random.Random(42)
hop_sequence = [rng.randint(1, 7) for _ in range(50)]

results = []
for i, n_hops in enumerate(hop_sequence):
    q1, q2 = qapi.create_qubits(2)
    qapi.assign_qstate([q1, q2], ks.b00)
    delta_time_ns = n_hops * HERALDING_DWELL_NS
    qapi.delay_depolarize(q2, depolar_rate, delta_time_ns)
    fid = float(qapi.fidelity([q1, q2], ks.b00, squared=True))
    results.append((i, n_hops, fid))
    qapi.discard(q1)
    qapi.discard(q2)  # cleanup — hàm test này CÓ discard
    if i < 10 or i > 44:
        print(f"{i:>4} {n_hops:>4} {fid:>10.4f}")

print("\n=== Kiểm tra: fidelity co giam dan theo hop khong (bo qua thu tu) ===")
by_hops = {}
for i, h, f in results:
    by_hops.setdefault(h, []).append(f)
for h in sorted(by_hops):
    vals = by_hops[h]
    print(f"hops={h}: fidelity trung binh={sum(vals)/len(vals):.4f}  "
          f"(n={len(vals)}, min={min(vals):.4f}, max={max(vals):.4f})")

print("\nKY VONG: fidelity giam dan khi hops tang (1-hop cao nhat, 7-hop thap nhat).")
print("NEU KHONG giam dan / gia tri lung tung -> xac nhan bug tich luy qubit.")
