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
    results_raw_<n>.csv       — 1 dòng/trial (100 dòng), ghi NGAY sau mỗi trial;
                                 chạy lại cùng lệnh sẽ bỏ qua các seed đã có
                                 (dừng/tiếp tục an toàn). Dùng --fresh để chạy lại từ đầu.
    results_summary_<n>.csv   — 1 dòng tổng hợp Mean±SD, ghi nối vào
                                 results_summary.csv chung cho cả 3 quy mô
"""

import argparse
import csv
import math
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from demo_scale_pipeline import run_scenario


def _coerce(v: str):
    """CSV text -> bool / float / None, so rows read back from the checkpoint
    file behave like the dicts returned by run_scenario()."""
    if v == "True":
        return True
    if v == "False":
        return False
    if v == "":
        return None
    try:
        return float(v)
    except ValueError:
        return v


def _load_checkpoint(raw_path: str):
    """Return (header, rows) from an existing raw file, or (None, [])."""
    if not Path(raw_path).exists():
        return None, []
    with open(raw_path, newline="") as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames
        rows = [{k: _coerce(v) for k, v in r.items()} for r in reader]
    return header, rows


def _tag(load: float, skip_qaoa: bool, exact_time_limit_s):
    """File-name suffix. Empty for the original setting, so default runs keep the
    original file names; any non-default option gets its own files and can never
    mix with (or overwrite) results from another configuration."""
    t = ""
    if load != 1.0:
        t += f"_L{load:g}"
    if skip_qaoa:
        t += "_noqaoa"
    if exact_time_limit_s is not None:
        t += f"_T{exact_time_limit_s:g}"
    return t


def run_100_trials(n_nodes: int, n_trials: int = 100, base_seed: int = 42,
                   fresh: bool = False, load: float = 1.0, skip_qaoa: bool = False,
                   exact_time_limit_s: float | None = None):
    tag = _tag(load, skip_qaoa, exact_time_limit_s)
    raw_path = f"results_raw_{n_nodes}{tag}.csv"

    # Checkpoint: each finished trial is appended to raw_path immediately, and a
    # re-run skips seeds already present, so the job can be stopped and resumed.
    if fresh and Path(raw_path).exists():
        backup = f"results_raw_{n_nodes}{tag}_old_{int(time.time())}.csv"
        Path(raw_path).rename(backup)
        print(f"  [--fresh] đã sao lưu file cũ thành {backup}")
    header, rows = _load_checkpoint(raw_path)
    required = {"trial_seed", "size_A", "U_x", "exact_U_x", "sa_repair_U_x",
                "exact_bound", "n_requests", "gap_ub_greedy_pct"}
    if header is not None and not required <= set(header):
        raise SystemExit(
            f"\n[!] {raw_path} được tạo bởi phiên bản code cũ (thiếu cột "
            f"{sorted(required - set(header))}).\n"
            f"    Chạy lại với --fresh để sao lưu file cũ và bắt đầu sạch.")
    wanted = set(range(base_seed, base_seed + n_trials))
    done = {int(r["trial_seed"]) for r in rows} & wanted
    if done:
        print(f"  Tiếp tục: đã có {len(done)}/{n_trials} trial trong {raw_path}, "
              f"chỉ chạy các seed còn thiếu.")
    fieldnames = header
    ran = 0

    t_start = time.perf_counter()
    for trial_idx in range(n_trials):
        seed = base_seed + trial_idx  # 42..141, đúng quy ước Table 5
        if seed in done:
            continue
        result = run_scenario(n_nodes, trial_seed=seed, verbose=False, load=load,
                              skip_qaoa=skip_qaoa, exact_time_limit_s=exact_time_limit_s)
        if "error" in result:
            print(f"  trial {trial_idx} (seed={seed}): SKIP — {result['error']}")
            continue

        if fieldnames is None:
            fieldnames = list(result.keys())
        elif list(result.keys()) != list(fieldnames):
            raise SystemExit(
                f"\n[!] Cột của {raw_path} khác với cột hiện tại của run_scenario() "
                f"(file được tạo bởi phiên bản code cũ).\n"
                f"    Chạy lại với --fresh để sao lưu file cũ và bắt đầu sạch.")
        new_file = not Path(raw_path).exists()
        with open(raw_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if new_file:
                writer.writeheader()
            writer.writerow(result)
            f.flush()
        rows.append(result)
        done.add(seed)
        ran += 1

        elapsed = time.perf_counter() - t_start
        avg = elapsed / ran
        remaining = avg * (n_trials - len(done))
        print(f"  [{len(done)}/{n_trials}] seed={seed}  "
              f"|A|={result['size_A']}  U(x)={result['U_x']:.2f}  "
              f"ILP={'opt' if result['exact_optimal'] else 'timeout'} {result['exact_runtime_s']:.1f}s  "
              f"ETA còn lại: {remaining/60:.1f} phút", end="\r")

    print()  # newline sau progress bar

    rows = sorted((r for r in rows if int(r["trial_seed"]) in wanted),
                  key=lambda r: int(r["trial_seed"]))
    if not rows:
        print("Không có trial nào thành công — kiểm tra lại lỗi.")
        return
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
                     "sa_repair_U_x", "tabu_repair_U_x", "ga_repair_U_x", "aco_repair_U_x",
                     "exact_U_x", "exact_runtime_s", "exact_bound", "n_requests",
                     "gap_qaoa_repair_pct", "gap_random_repair_pct", "gap_greedy_pct",
                     "gap_sa_repair_pct", "gap_tabu_repair_pct", "gap_ga_repair_pct",
                     "gap_aco_repair_pct",
                     "gap_ub_qaoa_repair_pct", "gap_ub_random_repair_pct",
                     "gap_ub_greedy_pct", "gap_ub_sa_repair_pct", "gap_ub_tabu_repair_pct",
                     "gap_ub_ga_repair_pct", "gap_ub_aco_repair_pct",
                     "peak_memory_mb"]
    summary = {"n_nodes": n_nodes, "load": load, "skip_qaoa": skip_qaoa,
               "exact_time_limit_s": exact_time_limit_s if exact_time_limit_s is not None else "default",
               "n_trials": len(rows)}
    for col in numeric_cols:
        vals = [r[col] for r in rows if col in r and r[col] is not None]
        vals = [float(v) for v in vals if math.isfinite(float(v))]  # gap=NaN khi ILP không tối ưu
        if not vals:
            continue
        summary[f"{col}_n"] = len(vals)
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
    for m in ("sa", "tabu", "ga", "aco"):
        c = sum(1 for r in rows if r.get(f"{m}_repair_feasible"))
        summary[f"{m}_repair_feasibility_rate_pct"] = 100.0 * c / len(rows)
    exact_ok = sum(1 for r in rows if r.get("exact_optimal"))
    summary["exact_optimal_rate_pct"] = 100.0 * exact_ok / len(rows)

    # ANOVA: QAOA+Repair, Random+Repair, Greedy, SA (cùng trial)
    try:
        from scipy import stats as scipy_stats
        groups = []
        # chỉ các phương pháp ĐÃ QUA Repair (nghiệm đều khả thi, so sánh được);
        # U thô của SA/Tabu/GA/ACO có thể là nghiệm vi phạm ràng buộc.
        for key in ([] if skip_qaoa else ["U_x"]) + ["random_repair_U_x", "greedy_U_x", "sa_repair_U_x",
                    "tabu_repair_U_x", "ga_repair_U_x", "aco_repair_U_x"]:
            vals = [r[key] for r in rows if key in r]
            if vals:
                groups.append([float(v) for v in vals])
        groups = [g for g in groups if all(math.isfinite(v) for v in g)]
        if len(groups) >= 2:
            f_stat, p_value = scipy_stats.f_oneway(*groups)
            summary["anova_F"] = f_stat
            summary["anova_p"] = p_value
    except ImportError:
        print("  (scipy không có sẵn — bỏ qua tính ANOVA, cần cài `pip install scipy`)")
        summary["anova_F"] = None
        summary["anova_p"] = None

    summary_path = f"results_summary{tag}.csv"
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
    if not skip_qaoa:
        print(f"  QAOA+Repair     = {summary['U_x_mean']:.2f} ± {summary['U_x_sd']:.2f}  "
              f"feas={summary['feasibility_rate_pct']:.1f}%")
    print(f"  Greedy          = {summary['greedy_U_x_mean']:.2f} ± {summary['greedy_U_x_sd']:.2f}  "
          f"feas={summary['greedy_feasibility_rate_pct']:.1f}%")
    print(f"  SA              = {summary['sa_U_x_mean']:.2f} ± {summary['sa_U_x_sd']:.2f}  "
          f"feas={summary['sa_feasibility_rate_pct']:.1f}%")
    if "exact_U_x_mean" in summary:
        print(f"  Exact (ILP)     = {summary['exact_U_x_mean']:.2f}  "
              f"(tối ưu được chứng minh: {summary['exact_optimal_rate_pct']:.0f}% trial)")
        for key, label in [("gap_qaoa_repair_pct", "QAOA+Repair"), ("gap_greedy_pct", "Greedy"),
                           ("gap_sa_repair_pct", "SA+Repair")]:
            if f"{key}_mean" in summary:
                print(f"  Gap {label:<12}= {summary[f'{key}_mean']:.3f}% ± {summary[f'{key}_sd']:.3f}"
                      f"  (n={summary[f'{key}_n']})")
        for key, label in [("gap_ub_greedy_pct", "Greedy"), ("gap_ub_tabu_repair_pct", "Tabu+Repair"),
                           ("gap_ub_random_repair_pct", "Random+Rep")]:
            if f"{key}_mean" in summary:
                print(f"  Gap_UB {label:<11}= {summary[f'{key}_mean']:.3f}% ± {summary[f'{key}_sd']:.3f}"
                      f"  (cận trên hợp lệ, n={summary[f'{key}_n']})")
    if summary.get("anova_F") is not None:
        print(f"  ANOVA (các phương pháp đã Repair): F={summary['anova_F']:.4f}  p={summary['anova_p']:.6f}")
    if not skip_qaoa:
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
    parser.add_argument("--fresh", action="store_true",
                        help="sao lưu results_raw_<n>.csv hiện có và chạy lại từ đầu")
    parser.add_argument("--load", type=float, default=1.0,
                        help="n_requests = round(load * n_nodes); 1.0 = cài đặt gốc")
    parser.add_argument("--skip-qaoa", action="store_true",
                        help="bỏ nhánh QAOA (Giai đoạn 1: chỉ baseline + ILP)")
    parser.add_argument("--exact-time-limit", type=float, default=None,
                        help="giới hạn thời gian ILP (s); mặc định 60")
    args = parser.parse_args()

    print(f"Bắt đầu {args.n_trials} trial cho n_nodes={args.n_nodes} "
          f"(seed {args.base_seed}-{args.base_seed + args.n_trials - 1})...")
    run_100_trials(args.n_nodes, args.n_trials, args.base_seed, args.fresh,
                   load=args.load, skip_qaoa=args.skip_qaoa,
                   exact_time_limit_s=args.exact_time_limit)
