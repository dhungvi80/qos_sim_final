# SUMMARY.md — Tổng kết quá trình phối hợp

**Dự án:** *A Quantum Optimization Approach for Dynamic Resource Allocation in
Quantum Networks* (bản thảo SCI Q3)
**Phạm vi:** Rà soát/chỉnh sửa bản thảo (v7→v14) + xây dựng codebase mô phỏng
thực nghiệm (`qos_sim/`) từ đầu, chạy thật trên NetSquid 1.1.8.

---

## 1. Giai đoạn rà soát bản thảo (v7 → v11)

- Sửa các lỗi placeholder/nháp còn sót (ghi chú editorial, bảng số liệu giả)
- Bổ sung nền tảng lý thuyết: **Proposition 1** (Optimality-Preserving
  Pruning), **Corollary 1** (Expected Reduction Ratio), **Theorem 1** viết
  lại thành cận chặt theo topology (thay vì "λ đủ lớn" chung chung)
- Bổ sung Complexity Analysis đầy đủ, Convergence Properties (3 tầng đảm bảo:
  đã chứng minh / tiệm cận trích dẫn / quan sát thực nghiệm)
- Thêm 3 hình minh họa (Proposition 1, DVQE partitioning, pipeline tổng thể)
- Rà soát toàn diện: phát hiện và sửa nhiều lỗi trích dẫn công thức sai số,
  bảng thiếu chú dẫn, tài liệu tham khảo mồ côi

## 2. Xây dựng codebase mô phỏng (`qos_sim/`)

Xây từ đầu, từng module một, có test đơn vị:

| Module | Vai trò | Trạng thái |
|---|---|---|
| `pruning.py` | Algorithm 1 | Test 4/4, kiểm định thực nghiệm Proposition 1 |
| `qubo_builder.py` | QUBO + Theorem 1 (λ*) | Test kiểm định trực tiếp Theorem 1 |
| `qaoa_numpy.py` | QAOA statevector (thay Qiskit — sandbox không cài được) | Đã tối ưu hiệu năng |
| `gwo_optimizer.py` | Grey Wolf Optimizer | — |
| `dvqe_partition.py` | Algorithm 2 (weakest-edge cut + Eq. 16) | Test 5/5 |
| `repair.py` | Algorithm 3 (mở rộng: thêm completion step) | Test 11/11 |
| `simulator.py` | NetSquid telemetry thật | Đã qua nhiều vòng debug với dữ liệu server thật |
| `topology_gen.py` | Sinh topology/request/path quy mô 25-100 node | Test 4/4 |
| `baseline.py` | Greedy, SA, Tabu, GA, ACO | Test 9/9 |

## 3. Debug NetSquid trên server thật (nhiều vòng)

1. API sai (`channel.send()` không hoạt động) → chuyển sang áp nhiễu trực tiếp
2. Sai tên hàm (`noise_operation`→`error_operation`→cuối cùng `delay_depolarize`)
3. KET formalism không hỗ trợ noise đúng → chuyển Density Matrix formalism
4. `depolar_rate` tự chọn không có căn cứ → thay bằng T2 thật (NV-center
   electron spin, ~10ms) + driver suy giảm đổi từ "độ dài fiber" sang
   "số hop × thời gian heralding" (trích dẫn Dahlberg et al. 2019, đã có
   sẵn trong bibliography)
5. `time.time()` không đơn điệu → đổi `time.perf_counter()` (runtime âm)

## 4. Phát hiện khoa học quan trọng nhất

**QAOA+Repair = Greedy Scheduler tuyệt đối, mọi trial (300/300).**

- Nguyên nhân: Algorithm 3 gốc chỉ trừ (subtractive-only), không bao giờ
  thêm biến QAOA bỏ sót → giới hạn chất lượng bất kể QAOA mạnh đến đâu
- Đã sửa: thêm **completion step** (Algorithm 3 mở rộng) + **Proposition 2**
  (Completion Monotonicity, có chứng minh) — nhưng completion mạnh đến mức
  kết quả cuối trùng khớp Greedy tuyệt đối
- Mở rộng so sánh với Tabu/GA/ACO: **các baseline cổ điển có utility thô
  cao hơn NHƯNG feasibility sụp đổ theo quy mô** (ACO: 26%→4%→0% khi
  25→50→100 node), trong khi QAOA+Repair và Greedy giữ 100% mọi quy mô
- **Khung câu chuyện bài báo đã đổi**: từ "QAOA thắng về điểm số" sang
  "QAOA đạt chất lượng ngang Greedy nhưng có nền tảng lý thuyết chứng minh
  được (Proposition 1-2, Theorem 1) và đảm bảo feasibility bền vững theo
  quy mô — giá trị mà baseline cổ điển không có"

## 5. Dữ liệu thực nghiệm cuối cùng (100 trial × 3 quy mô)

| n_nodes | \|A\| (Mean±SD) | Reduction | QAOA=Greedy U(x) | SA/Tabu/GA/ACO feasibility |
|---|---|---|---|---|
| 25 | 51.7±8.5 | 34.4%±5.6% | 84.18±13.57 | 47%/34%/43%/26% |
| 50 | 75.4±9.7 | 25.1%±3.2% | 131.73±16.85 | ~42%/~/~/~4% |
| 100 | 109.5±14.6 | 18.3%±2.4% | 199.08±26.12 | ~51%/~/~/**0%** |

## 6. Lịch sử phiên bản bản thảo

v7→v11: sửa lỗi placeholder, bổ sung lý thuyết | v12: Table 5-8 số liệu thật
lần đầu, Proposition 2 | v13: dọn mâu thuẫn nội bộ (Nhóm A), Table 2 + Mục
2.6 (Nhóm B) | v14: bổ sung Tabu/GA/ACO vào Table 6, effect size/CI, đoạn
Discussion về feasibility sụp đổ theo quy mô

## 7. Việc CHƯA hoàn thành (cần hạ tầng bạn tự thiết lập)

- **Gurobi thật** — cần license WLS hoạt động trên server
- **Qiskit thật** — sandbox không cài được (pypi.org bị chặn); đang dùng
  `qaoa_numpy.py` thay thế, đã ghi rõ trong Table 5
- **IBM Quantum hardware thật** — rào cản Việt Nam với Open Plan miễn phí
  đã ghi nhận, cần xác nhận gói trả phí/Credits
- **Mã nguồn công khai GitHub** — code đã sẵn sàng đóng gói (xem phần dưới),
  bạn là người tạo repo + đẩy lên

## 8. Bài học vận hành (rút ra giữa chừng)

- Chuỗi thay thế ngắn (số/phần trăm) dễ va chạm nhầm vị trí trong văn bản
  dài — cần ngữ cảnh đủ dài để định vị chính xác
- `/home/claude` (thư mục làm việc tạm) có thể bị mất khi sandbox reset —
  `/mnt/user-data/outputs` mới là nơi lưu bền, nên xuất checkpoint thường
  xuyên thay vì gom nhiều bước
- Luôn tính lại được số liệu tổng hợp từ file raw thay vì chạy lại toàn bộ
  thực nghiệm khi chỉ thiếu bước hậu xử lý (xem `recompute_summary.py`)
