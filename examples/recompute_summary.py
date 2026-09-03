"""
recompute_summary.py — tính lại results_summary.csv từ các file
results_raw_<n>.csv đã có sẵn, KHÔNG cần chạy lại 100-trial.

Dùng khi: file summary cũ thiếu cột (ví dụ mới thêm Tabu/GA/ACO/memory
vào demo_scale_pipeline.py sau khi đã chạy xong raw data).

Chạy (từ thư mục gốc dự án, nơi có sẵn results_raw_25.csv,
results_raw_50.csv, results_raw_100.csv):
    python3 examples/recompute_summary.py
"""

import csv
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

NUMERIC_COLS = [
    "size_D", "size_A", "reduction_ratio", "n_conflicts",
    "lambda_star", "n_partitions", "reconstructed_HC",
    "U_x", "total_weight", "runtime_s",
    "raw_qaoa_U_x", "greedy_U_x", "greedy_runtime_s",
    "sa_U_x", "sa_runtime_s",
    "tabu_U_x", "tabu_runtime_s",
    "ga_U_x", "ga_runtime_s",
    "aco_U_x", "aco_runtime_s",
    "peak_memory_mb",
]

FEASIBLE_COLS = ["feasible", "raw_qaoa_feasible", "greedy_feasible",
                  "sa_feasible", "tabu_feasible", "ga_feasible", "aco_feasible"]


def recompute(n_nodes: int) -> dict:
    raw_path = f"results_raw_{n_nodes}.csv"
    if not Path(raw_path).exists():
        print(f"  [!] {raw_path} không tồn tại, bỏ qua n_nodes={n_nodes}")
        return None

    with open(raw_path) as f:
        rows = list(csv.DictReader(f))
    print(f"  Đọc {len(rows)} trial từ {raw_path}")

    summary = {"n_nodes": n_nodes, "n_trials": len(rows)}
    for col in NUMERIC_COLS:
        vals = [float(r[col]) for r in rows if col in r and r[col] not in ("", None)]
        if not vals:
            print(f"    (cột '{col}' không có trong {raw_path} — bỏ qua)")
            continue
        summary[f"{col}_mean"] = statistics.mean(vals)
        summary[f"{col}_sd"] = statistics.stdev(vals) if len(vals) > 1 else 0.0

    for col in FEASIBLE_COLS:
        vals = [r[col] for r in rows if col in r]
        if not vals:
            continue
        # CSV lưu bool dạng chuỗi "True"/"False"
        count = sum(1 for v in vals if str(v).strip() == "True")
        key = col if col == "feasible" else col.replace("_feasible", "")
        summary[f"{key}_feasibility_rate_pct" if col != "feasible"
                 else "feasibility_rate_pct"] = 100.0 * count / len(vals)

    try:
        from scipy import stats as scipy_stats
        groups = []
        for key in ["U_x", "greedy_U_x", "sa_U_x", "tabu_U_x", "ga_U_x", "aco_U_x"]:
            vals = [float(r[key]) for r in rows if key in r and r[key] not in ("", None)]
            if vals:
                groups.append(vals)
        if len(groups) >= 2:
            f_stat, p_value = scipy_stats.f_oneway(*groups)
            summary["anova_F"] = f_stat
            summary["anova_p"] = p_value
    except ImportError:
        print("  (scipy không có sẵn — bỏ qua ANOVA)")

    return summary


def main():
    summaries = []
    for n in [25, 50, 100]:
        print(f"\n=== n_nodes={n} ===")
        s = recompute(n)
        if s is not None:
            summaries.append(s)

    if not summaries:
        print("Không có file raw nào tìm thấy — kiểm tra lại đường dẫn.")
        return

    out_path = "results_summary_recomputed.csv"
    fieldnames = sorted(set().union(*[s.keys() for s in summaries]))
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for s in summaries:
            writer.writerow(s)

    print(f"\nĐã ghi {len(summaries)} dòng tổng hợp vào {out_path}")
    for s in summaries:
        print(f"\nn_nodes={s['n_nodes']}:")
        for method in ["U_x", "greedy_U_x", "sa_U_x", "tabu_U_x", "ga_U_x", "aco_U_x"]:
            if f"{method}_mean" in s:
                print(f"  {method}: {s[f'{method}_mean']:.2f} ± {s[f'{method}_sd']:.2f}")
        if "peak_memory_mb_mean" in s:
            print(f"  peak_memory_mb: {s['peak_memory_mb_mean']:.1f} ± "
                  f"{s['peak_memory_mb_sd']:.1f}")


if __name__ == "__main__":
    main()
