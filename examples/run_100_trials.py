"""
run_100_trials.py — chạy 100 trial/quy mô (seed 42-141, đúng quy ước
Table 5), tổng hợp Mean±SD cho Table 6-8.

Chạy (mỗi quy mô riêng, có thể chạy song song trên nhiều terminal):
    python3 examples/run_100_trials.py --n-nodes 25
    python3 examples/run_100_trials.py --n-nodes 50
    python3 examples/run_100_trials.py --n-nodes 100

Ước tính thời gian: ~2-4s/trial x 100 trial = 5-10 phút/quy mô.
Khuyến nghị chạy trong `screen`/`tmux` để không bị ngắt giữa chừng:
    screen -S trial25
    python3 examples/run_100_trials.py --n-nodes 25
    (Ctrl+A rồi D để thoát screen, không dừng tiến trình)
    screen -r trial25   # quay lại xem tiến độ

Output:
    results_raw_<n>.csv       — 1 dòng/trial (100 dòng)
    results_summary_<n>.csv   — 1 dòng tổng hợp Mean±SD, ghi nối vào
                                 results_summary.csv chung cho cả 3 quy mô
"""

import argparse
import csv
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from demo_scale_pipeline import run_scenario


def run_100_trials(n_nodes: int, n_trials: int = 100, base_seed: int = 42):
    raw_path = f"results_raw_{n_nodes}.csv"
    fieldnames = None
    rows = []

    t_start = time.perf_counter()
    for trial_idx in range(n_trials):
        seed = base_seed + trial_idx  # 42..141, đúng quy ước Table 5
        result = run_scenario(n_nodes, trial_seed=seed, verbose=False)
        if "error" in result:
            print(f"  trial {trial_idx} (seed={seed}): SKIP — {result['error']}")
            continue
        rows.append(result)
        if fieldnames is None:
            fieldnames = list(result.keys())

        elapsed = time.perf_counter() - t_start
        avg = elapsed / (trial_idx + 1)
        remaining = avg * (n_trials - trial_idx - 1)
        print(f"  [{trial_idx+1}/{n_trials}] seed={seed}  "
              f"|A|={result['size_A']}  U(x)={result['U_x']:.2f}  "
              f"ETA còn lại: {remaining/60:.1f} phút", end="\r")

    print()  # newline sau progress bar

    if not rows:
        print("Không có trial nào thành công — kiểm tra lại lỗi.")
        return

    with open(raw_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Đã lưu {len(rows)} trial vào {raw_path}")

    # Tổng hợp Mean +/- SD cho các cột số quan trọng
    numeric_cols = ["size_D", "size_A", "reduction_ratio", "n_conflicts",
                     "lambda_star", "n_partitions", "reconstructed_HC",
                     "U_x", "total_weight", "runtime_s",
                     "raw_qaoa_U_x",
                     "random_repair_U_x", "random_repair_runtime_s",
                     "greedy_U_x", "greedy_runtime_s",
                     "sa_U_x", "sa_runtime_s",
                     "tabu_U_x", "tabu_runtime_s",
                     "ga_U_x", "ga_runtime_s",
                     "aco_U_x", "aco_runtime_s",
                     "peak_memory_mb"]
    summary = {"n_nodes": n_nodes, "n_trials": len(rows)}
    for col in numeric_cols:
        vals = [r[col] for r in rows if col in r]
        vals = [float(v) for v in vals]
        if not vals:
            continue
        summary[f"{col}_mean"] = statistics.mean(vals)
        summary[f"{col}_sd"] = statistics.stdev(vals) if len(vals) > 1 else 0.0

    feasible_count = sum(1 for r in rows if r.get("feasible"))
    summary["feasibility_rate_pct"] = 100.0 * feasible_count / len(rows)
    raw_feasible_count = sum(1 for r in rows if r.get("raw_qaoa_feasible"))
    summary["raw_qaoa_feasibility_rate_pct"] = 100.0 * raw_feasible_count / len(rows)
    random_feasible_count = sum(1 for r in rows if r.get("random_repair_feasible"))
    summary["random_repair_feasibility_rate_pct"] = (
        100.0 * random_feasible_count / len(rows)
    )
    greedy_feasible_count = sum(1 for r in rows if r.get("greedy_feasible"))
    summary["greedy_feasibility_rate_pct"] = 100.0 * greedy_feasible_count / len(rows)
    sa_feasible_count = sum(1 for r in rows if r.get("sa_feasible"))
    summary["sa_feasibility_rate_pct"] = 100.0 * sa_feasible_count / len(rows)
    tabu_feasible_count = sum(1 for r in rows if r.get("tabu_feasible"))
    summary["tabu_feasibility_rate_pct"] = 100.0 * tabu_feasible_count / len(rows)
    ga_feasible_count = sum(1 for r in rows if r.get("ga_feasible"))
    summary["ga_feasibility_rate_pct"] = 100.0 * ga_feasible_count / len(rows)
    aco_feasible_count = sum(1 for r in rows if r.get("aco_feasible"))
    summary["aco_feasibility_rate_pct"] = 100.0 * aco_feasible_count / len(rows)

    # ANOVA: QAOA+Repair, Random+Repair, Greedy, SA (cùng trial)
    try:
        from scipy import stats as scipy_stats
        groups = []
        for key in ["U_x", "random_repair_U_x", "greedy_U_x", "sa_U_x",
                    "tabu_U_x", "ga_U_x", "aco_U_x"]:
            vals = [r[key] for r in rows if key in r]
            if vals:
                groups.append([float(v) for v in vals])
        if len(groups) >= 2:
            f_stat, p_value = scipy_stats.f_oneway(*groups)
            summary["anova_F"] = f_stat
            summary["anova_p"] = p_value
    except ImportError:
        print("  (scipy không có sẵn — bỏ qua tính ANOVA, cần cài `pip install scipy`)")
        summary["anova_F"] = None
        summary["anova_p"] = None

    summary_path = "results_summary.csv"
    file_exists = Path(summary_path).exists()
    if file_exists:
        with open(summary_path, "r", newline="") as f:
            old_header = next(csv.reader(f), [])
        if old_header != list(summary.keys()):
            backup_path = f"results_summary_old_{int(time.time())}.csv"
            Path(summary_path).rename(backup_path)
            print(f"  [!] Cấu trúc cột đã đổi so với file cũ — đã sao lưu "
                  f"file cũ thành {backup_path}, tạo file mới sạch.")
            file_exists = False

    with open(summary_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(summary)

    print(f"\n=== Tổng hợp n_nodes={n_nodes} ({len(rows)} trial) ===")
    print(f"  |A| = {summary['size_A_mean']:.1f} ± {summary['size_A_sd']:.1f}")
    print(f"  reduction_ratio = {summary['reduction_ratio_mean']:.3f} ± "
          f"{summary['reduction_ratio_sd']:.3f}")
    print(f"  --- Ablation (Random+Repair vs QAOA+Repair vs Greedy) ---")
    print(f"  Random+Repair   = {summary.get('random_repair_U_x_mean', float('nan')):.2f} ± "
          f"{summary.get('random_repair_U_x_sd', 0):.2f}  "
          f"feas={summary.get('random_repair_feasibility_rate_pct', 0):.1f}%")
    print(f"  QAOA+Repair     = {summary['U_x_mean']:.2f} ± {summary['U_x_sd']:.2f}  "
          f"feas={summary['feasibility_rate_pct']:.1f}%")
    print(f"  Greedy          = {summary['greedy_U_x_mean']:.2f} ± {summary['greedy_U_x_sd']:.2f}  "
          f"feas={summary['greedy_feasibility_rate_pct']:.1f}%")
    print(f"  SA              = {summary['sa_U_x_mean']:.2f} ± {summary['sa_U_x_sd']:.2f}  "
          f"feas={summary['sa_feasibility_rate_pct']:.1f}%")
    if summary.get("anova_F") is not None:
        print(f"  ANOVA: F={summary['anova_F']:.4f}  p={summary['anova_p']:.6f}")
    print(f"  --- Raw QAOA (trước repair) ---")
    print(f"  Raw QAOA        = {summary['raw_qaoa_U_x_mean']:.2f} ± "
          f"{summary['raw_qaoa_U_x_sd']:.2f}  "
          f"feas={summary['raw_qaoa_feasibility_rate_pct']:.1f}%")
    print(f"  runtime = {summary['runtime_s_mean']:.2f}s ± {summary['runtime_s_sd']:.2f}s "
          f"per trial")
    print(f"  Đã ghi vào {summary_path}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-nodes", type=int, required=True, choices=[25, 50, 100])
    parser.add_argument("--n-trials", type=int, default=100)
    parser.add_argument("--base-seed", type=int, default=42)
    args = parser.parse_args()

    print(f"Bắt đầu {args.n_trials} trial cho n_nodes={args.n_nodes} "
          f"(seed {args.base_seed}-{args.base_seed + args.n_trials - 1})...")
    run_100_trials(args.n_nodes, args.n_trials, args.base_seed)
