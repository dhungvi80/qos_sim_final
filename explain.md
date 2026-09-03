# Explain — Giai đoạn: Tại sao tăng GWO/QAOA không cải thiện kết quả

## Hiện tượng
So sánh trước/sau khi tăng cấu hình (p=2→3, 15→30 wolves, 30→80 iterations):

| | Trước | Sau | Đổi |
|---|---|---|---|
| QAOA U(x), n=25 | 66.86±16.84 | 66.81±16.17 | ~0% |
| Raw QAOA feasibility, n=25 | 14% | 33% | +19pp |
| Raw QAOA feasibility, n=100 | 0% | 4% | +4pp |

Raw QAOA (trước repair) **có cải thiện rõ** (feasibility tăng đáng kể), nhưng
U(x) **sau** Algorithm 3 (Repair) gần như không đổi.

## Nguyên nhân gốc rễ

**Algorithm 3 (Repair) chỉ có thể XÓA bớt biến, không bao giờ THÊM biến mới.**

Cơ chế repair hiện tại: với mỗi cặp xung đột, giữ biến có priority cao hơn,
xóa biến kia. Đây là phép toán **chỉ trừ (subtractive-only)** trên x_raw:
- Nếu QAOA quên không bật 1 biến quan trọng (set x_i=0 thay vì 1), repair
  **không có cách nào bật lại** — nó chỉ xử lý những bit QAOA đã bật sẵn.
- Ngược lại, Greedy xây dựng lời giải từ đầu theo đúng thứ tự priority,
  nên luôn "vét" được mọi biến có lợi không xung đột — không bỏ sót.

Đây là lý do cấu trúc (không phải do QAOA "chưa đủ mạnh") khiến:
1. Tăng tài nguyên QAOA/GWO cải thiện raw feasibility nhưng không cải
   thiện U(x) cuối — vì repair vẫn chỉ "dọn dẹp" phần QAOA đã chọn, không
   bổ sung phần QAOA bỏ sót.
2. QAOA+Repair khó vượt Greedy về mặt cấu trúc — Greedy không có giới hạn
   "chỉ trừ" này.

## Ý nghĩa cho bài báo
Đây là **phát hiện khoa học thật, có giá trị** để viết vào Discussion —
không phải điểm yếu cần giấu. Điểm mạnh thật sự của phương pháp đề xuất
không nằm ở việc "thắng Greedy về utility" mà ở:
- Đảm bảo feasibility 100% tuyệt đối (Greedy cũng 100% nhưng SA chỉ 42-51%)
- Nền tảng lý thuyết (Proposition 1, Theorem 1) cho một pipeline lượng tử
  có thể mở rộng, trong khi Greedy là heuristic cổ điển không có gì để
  chứng minh/mở rộng thêm

## Lựa chọn hướng đi tiếp theo (cần quyết định)
A. Sửa Algorithm 3 để cho phép "thêm bù" biến bị bỏ sót sau khi xóa xung
   đột (ví dụ: sau bước xóa, quét lại các biến chưa dùng, thêm nếu không
   xung đột với tập hiện tại — về bản chất là lai Greedy vào cuối Repair)
   → có thể cải thiện U(x), nhưng thay đổi cách Algorithm 3 hoạt động,
   cần viết lại Theorem 1/proof liên quan nếu có ảnh hưởng.
B. Giữ nguyên, viết trung thực vào Discussion: đóng góp chính là feasibility
   guarantee + nền tảng lý thuyết, không phải vượt trội về utility thô.
C. Kết hợp: thêm bước "bù thêm" như (A) NHƯNG framing rõ đây là Algorithm 3
   phiên bản mở rộng (Algorithm 3b), giữ nguyên Algorithm 3 gốc cho phần
   lý thuyết đã chứng minh.

---

# Explain — Giai đoạn: Sửa Algorithm 3 (thêm Completion Step)

## Thiết kế
Algorithm 3 mở rộng thêm Bước 3 (Completion): sau khi giải quyết xung đột
(Bước 1, giữ nguyên), quét các biến CHƯA active theo thứ tự trọng số giảm
dần, thêm lại nếu không xung đột với tập hiện tại.

**Điểm mấu chốt:** Bước 3 CHỈ THÊM, không bao giờ XÓA — nên utility sau
completion luôn ≥ utility trước completion (mọi w_i > 0).

## Proposition 2 (đề xuất thêm vào bài, chưa đưa vào docx)

**Proposition 2 (Completion Monotonicity).** Let x₁ be the output of
Algorithm 3's Step 1 (conflict resolution) and x₂ the output after Step 3
(completion). Then U(x₂) ≥ U(x₁), and Algorithm 3 (extended) still
terminates in O(n log n) (Step 3 adds one sort O(n log n) + one linear
scan with bounded-degree conflict lookup, same complexity class as Step 1).

*Proof.* Step 3 only sets x_i: 0→1 for variables not conflicting with the
current active set; it never sets any x_i: 1→0. Since U(x) = Σ w_i x_i
with all w_i > 0 (Section 3.1), each such flip strictly increases U(x) or
leaves it unchanged (never decreases). Feasibility is preserved by
construction — a variable is only added if adjacency ∩ active = ∅, so no
new conflict is introduced. □

## Ảnh hưởng tới Theorem 1
**KHÔNG cần viết lại.** Theorem 1 chứng minh tính chất của cực tiểu toàn
cục hàm QUBO H(x) (Eq. 13) dưới điều kiện λ ≥ λ* — đây là tính chất của
BẢN THÂN hàm mục tiêu, độc lập với cách Algorithm 3 tìm/sửa nghiệm. Bước
1 (tie-break theo Δc) vẫn dùng nguyên logic cũ, nên vẫn đúng theo Theorem 1
như trước. Chỉ có "mức độ hoàn chỉnh" của heuristic thay đổi, không phải
tính đúng đắn của giới hạn lý thuyết.

## Kết quả dry-run (dữ liệu giả lập, chưa phải NetSquid thật)
QAOA+Repair (mở rộng) = 111.15, Greedy = 111.15 — **bằng nhau tuyệt đối**
(tỷ lệ 1.0). Đúng như Proposition 2 dự đoán về mặt cấu trúc: completion
step khiến QAOA+Repair hội tụ về ít nhất bằng Greedy.

## Lập luận lý thuyết vs thực nghiệm (cho Discussion)
- **Trước fix**: lý thuyết (Theorem 1, Proposition 1) đúng nhưng thực
  nghiệm cho kết quả utility thấp hơn Greedy — khoảng cách này đến từ
  heuristic Algorithm 3 gốc (chỉ trừ), không phải từ QUBO/QAOA sai.
- **Sau fix**: Proposition 2 dự đoán đúng hướng (không giảm), thực nghiệm
  dry-run xác nhận bằng số. Đây là ví dụ tốt cho bài báo về việc lý
  thuyết định hướng đúng cách sửa, thực nghiệm xác nhận định lượng.
- **Câu hỏi mở cần 100-trial thật trả lời**: QAOA+Repair mở rộng có VƯỢT
  Greedy ở một số trial (nhờ seed lượng tử khác biệt) hay chỉ luôn BẰNG?
  Nếu chỉ luôn bằng, giá trị của thành phần lượng tử cần được diễn giải
  lại (ví dụ: cùng chất lượng nhưng có nền tảng lý thuyết mở rộng được,
  không chỉ là "thắng điểm số").

---

# Explain — Giai đoạn: 100-trial thật xác nhận QAOA = Greedy tuyệt đối

## Kết quả
Không chỉ "≥ Greedy" như Proposition 2 dự đoán — QAOA+Repair (mở rộng)
**bằng tuyệt đối Greedy trên cả 300/300 trial** (3 quy mô × 100 trial),
sai số 0.00000. ANOVA giờ mất ý nghĩa thống kê ở n=100 (p=0.355).

## Nguyên nhân
Completion step (Bước 3) quét biến còn lại theo ĐÚNG thứ tự ưu tiên mà
Greedy dùng — nên bất kể QAOA (Bước 1) chọn gì trước đó, Bước 3 luôn "vá"
đủ để kết quả cuối trùng khớp với việc chạy Greedy từ đầu. Completion đang
"quá mạnh", che lấp hoàn toàn đóng góp của QAOA thay vì chỉ bổ trợ.

## 2 hướng xử lý

**Hướng 1 — Làm yếu completion, chỉ bổ trợ cục bộ:**
Giới hạn completion chỉ được thêm tối đa k biến (ví dụ k=1-2), hoặc chỉ
áp dụng completion cho các request mà QAOA "gần" chọn được (xác suất cao
trong phân phối QAOA nhưng không phải argmax) — giữ vai trò chính cho
QAOA, completion chỉ vá lỗi nhỏ thay vì tái tạo toàn bộ lời giải.

**Hướng 2 — Chấp nhận kết quả, đổi khung câu chuyện bài báo:**
Báo cáo trung thực: QAOA+Repair đạt CHẤT LƯỢNG NGANG BẰNG Greedy (không
vượt trội), nhưng có nền tảng lý thuyết chứng minh được (Proposition 1,
2, Theorem 1) mà Greedy không có — đóng góp của bài chuyển từ "thắng về
điểm số" sang "chứng minh được tại sao đạt ngang bằng, có thể mở rộng
lý thuyết cho các bài toán phức tạp hơn Greedy không giải quyết được".

## Khuyến nghị
Hướng 1 rủi ro cao hơn (có thể lại thua Greedy như trước khi fix, mất
thời gian dò lại tham số k). Hướng 2 an toàn, trung thực, và thực ra là
một câu chuyện khoa học hợp lý — nhiều bài QAOA thật cũng kết luận tương
tự (đạt ngang bằng, không nhất thiết vượt trội cổ điển ở quy mô nhỏ).
