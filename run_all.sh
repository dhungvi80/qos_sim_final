#!/usr/bin/env bash
cd "$(dirname "$0")"
for n in 25 50 100; do
  nohup python3 -u examples/run_100_trials.py --n-nodes $n > log_$n.txt 2>&1 &
done
start=$(date +%s)
while pgrep -f run_100_trials > /dev/null; do
  clear
  echo "Đã chạy: $(( ($(date +%s)-start)/60 )) phút $(( ($(date +%s)-start)%60 )) giây   (Ctrl+C để thoát bảng, tiến trình vẫn chạy)"
  echo "------------------------------------------------"
  for n in 25 50 100; do
    line=$(tail -c 400 log_$n.txt | tr '\r' '\n' | grep "ETA" | tail -1)
    if [ -n "$line" ]; then
      echo "$line" | sed -E "s/.*\[([0-9]+\/[0-9]+)\].*ETA còn lại: ([0-9.]+) phút.*/$n node: \1  còn ~\2 phút/"
    elif grep -q "Đã lưu" log_$n.txt; then
      echo "$n node: XONG"
    else
      echo "$n node: đang khởi động..."
    fi
  done
  sleep 5
done
echo; echo "=== HOÀN TẤT sau $(( ($(date +%s)-start)/60 )) phút ==="
ls -la results_raw_*.csv results_summary.csv
