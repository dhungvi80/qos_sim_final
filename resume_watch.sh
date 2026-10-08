#!/usr/bin/env bash
cd "$(dirname "$0")"
SCALES="25 50 100"

show() {
  for n in $SCALES; do
    if [ -f results_raw_$n.csv ]; then
      echo "$n node: XONG ($(($(wc -l < results_raw_$n.csv)-1)) trial trong results_raw_$n.csv)"
    elif [ -f log_$n.txt ] && tail -c 400 log_$n.txt | tr '\r' '\n' | grep -q ETA; then
      tail -c 400 log_$n.txt | tr '\r' '\n' | grep ETA | tail -1 \
        | sed -E "s/.*\[([0-9]+\/[0-9]+)\].*ETA còn lại: ([0-9.]+) phút.*/$n node: \1  còn ~\2 phút/"
    else
      echo "$n node: chưa có tiến độ (chưa chạy, hoặc log trống)"
    fi
  done
}

echo "=== TRẠNG THÁI HIỆN TẠI ==="
ps -o pid,stat,etime,cmd -C python3 | grep run_100_trials || echo "(không có tiến trình run_100_trials nào đang tồn tại)"
show
echo

# 1) Có tiến trình đang bị dừng (STAT chứa T) -> tiếp tục
if ps -o stat=,cmd= -C python3 | grep run_100_trials | grep -q '^T'; then
  echo ">> Phát hiện tiến trình đang tạm dừng, tiếp tục chạy..."
  pkill -CONT -f run_100_trials
fi

# 2) Không còn tiến trình nào nhưng còn quy mô chưa có results_raw -> khởi động lại quy mô đó
if ! pgrep -f run_100_trials > /dev/null; then
  todo=""
  for n in $SCALES; do [ -f results_raw_$n.csv ] || todo="$todo $n"; done
  if [ -n "$todo" ]; then
    echo ">> Tiến trình đã mất. Quy mô chưa xong:$todo (không có checkpoint nên chạy lại từ trial đầu)."
    read -p "Chạy lại các quy mô này? [y/N] " a
    [ "$a" = "y" ] || exit 0
    for n in $todo; do
      nohup python3 -u examples/run_100_trials.py --n-nodes $n > log_$n.txt 2>&1 &
    done
  else
    echo ">> Cả 3 quy mô đã có results_raw_*.csv. Không còn gì để chạy."; exit 0
  fi
fi

# 3) Bảng đếm ngược
start=$(date +%s)
while pgrep -f run_100_trials > /dev/null; do
  clear
  e=$(( $(date +%s)-start ))
  echo "Theo dõi: $((e/60)) phút $((e%60)) giây   (Ctrl+C chỉ thoát bảng, tiến trình vẫn chạy)"
  echo "------------------------------------------------"
  show
  sleep 5
done
echo; echo "=== HOÀN TẤT ==="; ls -la results_raw_*.csv results_summary.csv 2>/dev/null
