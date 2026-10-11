#!/usr/bin/env bash
# run_t3_parallel.sh — T3: chạy song song 100 trial × 3 quy mô (NetSquid)
# COLLISION_PRUNING=False (đã set trong demo_scale_pipeline.py)
#
# Dùng:
#   source .venv/bin/activate
#   bash run_t3_parallel.sh          # chạy mới (--fresh, sao lưu raw cũ)
#   bash run_t3_parallel.sh resume   # tiếp tục nếu bị ngắt (không --fresh)
#
set -euo pipefail
cd "$(dirname "$0")"

SCALES="25 50 100"
MODE="${1:-fresh}"   # fresh | resume

echo "=== T3 parallel  mode=$MODE  $(date '+%Y-%m-%d %H:%M:%S') ==="
echo "COLLISION_PRUNING phải = False trong examples/demo_scale_pipeline.py"
grep -n "COLLISION_PRUNING" examples/demo_scale_pipeline.py || true
echo

# --- khởi động 3 tiến trình ---
for n in $SCALES; do
  if pgrep -f "run_100_trials.py --n-nodes $n" > /dev/null; then
    echo "$n node: đã đang chạy — bỏ qua khởi động"
    continue
  fi
  if [ "$MODE" = "fresh" ]; then
    nohup python3 -u examples/run_100_trials.py --n-nodes "$n" --fresh \
      > "log_${n}.txt" 2>&1 &
    echo "$n node: started PID=$!  log=log_${n}.txt  (--fresh)"
  else
    nohup python3 -u examples/run_100_trials.py --n-nodes "$n" \
      > "log_${n}.txt" 2>&1 &
    echo "$n node: started PID=$!  log=log_${n}.txt  (resume)"
  fi
done

sleep 2

# --- bảng giám sát ---
start=$(date +%s)
while pgrep -f "run_100_trials.py" > /dev/null 2>&1; do
  clear
  elapsed=$(( $(date +%s) - start ))
  echo "T3 monitor  đã chạy: $((elapsed/60)) phút $((elapsed%60)) giây"
  echo "Ctrl+C chỉ thoát bảng — 3 tiến trình vẫn chạy nền"
  echo "--------------------------------------------------------------"
  printf "%-6s  %-12s  %-28s  %s\n" "scale" "status" "progress" "detail"
  echo "--------------------------------------------------------------"
  for n in $SCALES; do
    if pgrep -f "run_100_trials.py --n-nodes $n" > /dev/null 2>&1; then
      status="RUNNING"
    else
      status="DONE"
    fi
    # progress từ log (dòng ETA cuối)
    if [ -f "log_${n}.txt" ]; then
      eta_line=$(tr '\r' '\n' < "log_${n}.txt" | grep -E "ETA còn lại|Đã lưu" | tail -1 || true)
      if echo "$eta_line" | grep -q "Đã lưu"; then
        prog="100/100"
        detail="XONG — $eta_line"
      elif echo "$eta_line" | grep -q "ETA"; then
        prog=$(echo "$eta_line" | sed -E 's/.*\[([0-9]+\/[0-9]+)\].*/\1/')
        remain=$(echo "$eta_line" | sed -E 's/.*ETA còn lại: ([0-9.]+) phút.*/\1/')
        detail="còn ~${remain} phút"
      else
        nlines=$(wc -l < "log_${n}.txt" 2>/dev/null || echo 0)
        prog="..."
        detail="khởi động (log ${nlines} dòng)"
      fi
    else
      prog="—"
      detail="chưa có log"
    fi
    # số trial đã ghi
    raw="results_raw_${n}.csv"
    if [ -f "$raw" ]; then
      ntrial=$(( $(wc -l < "$raw") - 1 ))
      detail="$detail  | raw=${ntrial} trial"
    fi
    printf "%-6s  %-12s  %-28s  %s\n" "${n}" "$status" "$prog" "$detail"
  done
  echo "--------------------------------------------------------------"
  echo "log tail:"
  for n in $SCALES; do
    last=$(tr '\r' '\n' < "log_${n}.txt" 2>/dev/null | grep -E "\[.*\/.*\] seed=" | tail -1 || true)
    [ -n "$last" ] && echo "  $n: $last"
  done
  sleep 10
done

echo
echo "=== HOÀN TẤT sau $(( ($(date +%s)-start)/60 )) phút $(( ($(date +%s)-start)%60 )) giây ==="
echo
ls -la results_raw_25.csv results_raw_50.csv results_raw_100.csv 2>/dev/null || true
echo
echo "Kiểm tra nhanh ρ (reduction = 1 - A/D):"
python3 - <<'PY'
import csv
for n in [25, 50, 100]:
    path = f"results_raw_{n}.csv"
    try:
        rows = list(csv.DictReader(open(path)))
    except FileNotFoundError:
        print(f"  {n}: thiếu file")
        continue
    if not rows:
        print(f"  {n}: rỗng")
        continue
    D = [float(r["size_D"]) for r in rows]
    A = [float(r["size_A"]) for r in rows]
    rho = [1 - a/d for a, d in zip(A, D)]
    ra = sum(int(float(r.get("rejected_availability", 0))) for r in rows)
    rc = sum(int(float(r.get("rejected_collision", 0))) for r in rows)
    print(f"  n={n}: trials={len(rows)}  rho_mean={sum(rho)/len(rho):.4f}  "
          f"|A|={sum(A)/len(A):.1f}  avail_rej_total={ra}  coll_rej_total={rc}")
PY
echo
echo "Gửi Summary trên (hoặc 3 file results_raw_*.csv) để tiếp tục."
