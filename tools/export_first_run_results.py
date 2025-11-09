# -*- coding: utf-8 -*-
"""
export_first_run_results.py

读取第一次复现在 Result/Experiment1 下生成的所有 .pkl，
导出逐图 CSV、按 place 汇总 CSV，并生成一份“重放”日志，
在样式上尽量贴近你第一次运行 baseline.py 时看到的关键信息：
  Region | Place | Image | Progress ... + pred_error

默认扫描目录：
  D:\Acode\Benchmark-main\Result\Experiment1

输出文件（写到 root 目录下）：
  first_run_pred_errors.csv
  first_run_pred_errors_summary.csv
  first_run_replay_log.txt
"""
import os
import re
import csv
import math
import argparse
import pickle
from collections import defaultdict

def natural_key(s: str):
    """按人类直觉排序文件名（把数字部分按数值排序）"""
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]

def find_first_float(obj, key_predicates):
    """
    在任意嵌套结构(dict/list/tuple)中，寻找第一个满足 key_predicates(k) 的键对应的 float。
    """
    try:
        # dict
        if isinstance(obj, dict):
            # 先直接匹配 key
            for k, v in obj.items():
                try:
                    if isinstance(k, str) and key_predicates(k) and isinstance(v, (int, float)) and math.isfinite(float(v)):
                        return float(v)
                except Exception:
                    pass
            # 再递归 value
            for v in obj.values():
                out = find_first_float(v, key_predicates)
                if out is not None:
                    return out
        # list/tuple
        elif isinstance(obj, (list, tuple)):
            for v in obj:
                out = find_first_float(v, key_predicates)
                if out is not None:
                    return out
    except Exception:
        pass
    return None

def extract_pred_error(data):
    """
    尽可能鲁棒地从 .pkl 内容中提取 pred_error。
    常见命名：pred_error / pose_error / position_error / predErr 等。
    """
    def is_pred_error_key(k: str) -> bool:
        lk = k.lower()
        return (
            ("pred" in lk and "err" in lk) or
            ("pose" in lk and "err" in lk) or
            ("position" in lk and "err" in lk) or
            lk in {"pe", "pred_error"}
        )
    val = find_first_float(data, is_pred_error_key)
    return val

def extract_match_inlier(data):
    """
    试图从 .pkl 中提取匹配数/内点数，若没有就返回 (None, None)。
    常见命名：match_num/matches/num_matches, inlier_num/inliers/num_inliers。
    """
    def is_match_key(k: str) -> bool:
        lk = k.lower()
        return ("match" in lk and ("num" in lk or "count" in lk or lk.endswith("es")))
    def is_inlier_key(k: str) -> bool:
        lk = k.lower()
        return ("inlier" in lk and ("num" in lk or "count" in lk or lk.endswith("s")))

    m = find_first_float(data, is_match_key)
    i = find_first_float(data, is_inlier_key)
    # 有些实现把它们存成 int；如果是 float 也无妨
    if m is not None:
        m = int(round(m))
    if i is not None:
        i = int(round(i))
    return m, i

def derive_image_name_from_pkl(pkl_filename: str) -> str:
    """
    从 VG_data_*.pkl 反推原图像名（.JPG）。
    例：
      VG_data_DJI_0532.pkl -> DJI_0532.JPG
      VG_data_DJI_20240917112525_0413_D.pkl -> DJI_20240917112525_0413_D.JPG
    """
    name = os.path.splitext(os.path.basename(pkl_filename))[0]  # VG_data_DJI_0532
    if name.lower().startswith("vg_data_"):
        stem = name[len("vg_data_"):]
    else:
        stem = name
    return f"{stem}.JPG"

def scan_all_pkls(root_dir: str):
    """
    遍历 Experiment1 下的结构：
      <root>/<region>/pkl_<place>/resize_<ratio>/<method>/*.pkl
    返回条目列表，每条为 dict：
      {
        region, place, resize, method, pkl_path, image
      }
    """
    items = []
    if not os.path.isdir(root_dir):
        raise FileNotFoundError(f"root_dir not found: {root_dir}")

    for region in sorted(os.listdir(root_dir), key=natural_key):
        region_dir = os.path.join(root_dir, region)
        if not os.path.isdir(region_dir):
            continue

        for pkl_dir in sorted([d for d in os.listdir(region_dir) if d.lower().startswith("pkl_")], key=natural_key):
            place = pkl_dir[len("pkl_"):]
            place_dir = os.path.join(region_dir, pkl_dir)
            if not os.path.isdir(place_dir):
                continue

            for resize_dir in sorted([d for d in os.listdir(place_dir) if d.lower().startswith("resize_")], key=natural_key):
                resize = resize_dir[len("resize_"):]
                resize_path = os.path.join(place_dir, resize_dir)
                if not os.path.isdir(resize_path):
                    continue

                for method in sorted(os.listdir(resize_path), key=natural_key):
                    method_path = os.path.join(resize_path, method)
                    if not os.path.isdir(method_path):
                        continue

                    # 只收集 .pkl
                    pkl_files = [f for f in os.listdir(method_path) if f.lower().endswith(".pkl")]
                    for fname in sorted(pkl_files, key=natural_key):
                        pkl_path = os.path.join(method_path, fname)
                        image = derive_image_name_from_pkl(fname)
                        items.append(dict(
                            region=region,
                            place=place,
                            resize=resize,
                            method=method,
                            pkl_path=pkl_path,
                            image=image
                        ))
    return items

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        default=r"D:\Acode\Benchmark-main\Result\Experiment1",
        help="Experiment1 目录（包含 QZ_Town 等子目录）"
    )
    args = parser.parse_args()
    root_dir = args.root

    entries = scan_all_pkls(root_dir)
    if not entries:
        print(f"[WARN] 在 {root_dir} 没有发现任何 .pkl。请确认路径正确。")
        return

    # 逐图导出 CSV
    per_image_csv = os.path.join(root_dir, "first_run_pred_errors.csv")
    with open(per_image_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["region", "place", "resize", "method", "image", "pred_error", "matches", "inliers", "inlier_ratio", "pkl_path"])
        load_errors = 0

        # 进度重放 + 日志
        replay_log = os.path.join(root_dir, "first_run_replay_log.txt")
        log_lines = []
        n_total = len(entries)

        # 为了让输出阅读性更好，按 region -> place -> image 排序
        entries_sorted = sorted(entries, key=lambda x: (x["region"], x["place"], natural_key(x["image"])))

        for idx, e in enumerate(entries_sorted, 1):
            p = e["pkl_path"]
            try:
                with open(p, "rb") as fp:
                    data = pickle.load(fp)
            except Exception as ex:
                pred = None
                mnum = None
                inum = None
                load_errors += 1
                # 写 CSV（标记无法读取）
                writer.writerow([e["region"], e["place"], e["resize"], e["method"], e["image"], "", "", "", "", p])
                # 日志
                progress = (idx - 1) / n_total * 100.0
                log_lines.append(f"Region: {e['region']} | Place: {e['place']} | Image: {e['image']} | Progress: {progress:.1f}%")
                log_lines.append(f"pred_error: <LOAD_FAILED: {ex}>")
                continue

            pred = extract_pred_error(data)
            mnum, inum = extract_match_inlier(data)

            ratio = ""
            if mnum and inum is not None and mnum > 0:
                ratio = f"{inum / mnum:.6f}"

            writer.writerow([
                e["region"], e["place"], e["resize"], e["method"], e["image"],
                f"{pred:.15g}" if isinstance(pred, (int, float)) else "",
                mnum if mnum is not None else "",
                inum if inum is not None else "",
                ratio,
                p
            ])

            # ---- 重放日志（贴近第一次输出的关键信息）----
            progress = (idx - 1) / n_total * 100.0
            log_lines.append(f"Region: {e['region']} | Place: {e['place']} | Image: {e['image']} | Progress: {progress:.1f}%")
            # 如果能拿到匹配/内点，就也给出一行简要信息（第一次你的终端有一堆小数，我用更直观的“匹配/内点/占比”替代）
            if mnum is not None and inum is not None and mnum > 0:
                log_lines.append(f"matches: {mnum} | inliers: {inum} | inlier_ratio: {inum/mnum:.6f}")
            # 关键：pred_error
            if isinstance(pred, (int, float)):
                log_lines.append(f"pred_error: {pred}")
            else:
                log_lines.append("pred_error: <NOT_FOUND>")

        # 写日志文件
        with open(replay_log, "w", encoding="utf-8") as lf:
            lf.write("\n".join(log_lines) + "\n")

        print(f"[OK] 逐图 CSV 已写入: {per_image_csv}")
        print(f"[OK] 重放日志已写入: {replay_log}")
        if load_errors:
            print(f"[WARN] 有 {load_errors} 个 .pkl 读取失败，已在 CSV/日志中标注。")

    # 汇总（按 place 聚合）
    summary = defaultdict(list)
    with open(per_image_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if not row["pred_error"]:
                continue
            key = (row["region"], row["place"], row["resize"], row["method"])
            try:
                pe = float(row["pred_error"])
            except Exception:
                continue
            summary[key].append(pe)

    summary_csv = os.path.join(root_dir, "first_run_pred_errors_summary.csv")
    with open(summary_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["region", "place", "resize", "method", "count", "mean", "median", "min", "max"])
        for (region, place, resize, method), arr in sorted(summary.items()):
            arr_sorted = sorted(arr)
            n = len(arr_sorted)
            mean = sum(arr_sorted) / n if n else float("nan")
            med = (arr_sorted[n//2] if n % 2 == 1 else (arr_sorted[n//2 - 1] + arr_sorted[n//2]) / 2) if n else float("nan")
            mn = arr_sorted[0] if n else float("nan")
            mx = arr_sorted[-1] if n else float("nan")
            writer.writerow([region, place, resize, method, n, f"{mean:.6f}", f"{med:.6f}", f"{mn:.6f}", f"{mx:.6f}"])

    print(f"[OK] 汇总 CSV 已写入: {summary_csv}")

if __name__ == "__main__":
    main()
