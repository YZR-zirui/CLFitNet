#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import csv
import argparse
import numpy as np
import matplotlib.pyplot as plt

TIME_MIN_LIST = [0, 30, 60, 120, 240]
DIGIT_TO_MIN = {"2": 0, "3": 30, "4": 60, "5": 120, "6": 240}
IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def detect_split_roots(dataset_root: str, use_splits: str):
    has_train = os.path.isdir(os.path.join(dataset_root, "train"))
    has_test = os.path.isdir(os.path.join(dataset_root, "test"))
    has_valid = os.path.isdir(os.path.join(dataset_root, "valid")) or os.path.isdir(os.path.join(dataset_root, "val"))

    if has_train and has_test and has_valid:
        split_map = {
            "train": os.path.join(dataset_root, "train"),
            "test": os.path.join(dataset_root, "test"),
            "valid": os.path.join(dataset_root, "valid") if os.path.isdir(os.path.join(dataset_root, "valid")) else os.path.join(dataset_root, "val"),
        }
        if use_splits == "all":
            return [split_map["train"], split_map["valid"], split_map["test"]]
        if use_splits not in split_map:
            raise ValueError(f"--use_splits 只能是 all/train/valid/test, 但你给的是 {use_splits}")
        return [split_map[use_splits]]
    else:
        return [dataset_root]


def list_subject_dirs(root_dir: str):
    return sorted([
        os.path.join(root_dir, d)
        for d in os.listdir(root_dir)
        if os.path.isdir(os.path.join(root_dir, d))
    ])


def find_time_dir(eye_dir: str, tmin: int):
    target_digit = None
    for k, v in DIGIT_TO_MIN.items():
        if v == tmin:
            target_digit = k
            break
    if target_digit is None or (not os.path.isdir(eye_dir)):
        return None

    for d in os.listdir(eye_dir):
        p = os.path.join(eye_dir, d)
        if not os.path.isdir(p):
            continue
        head = d.strip().split(".")[0].strip()
        if head == target_digit:
            return p
    return None


def find_scan_new_dir(time_dir: str, scan_id: int = 1):
    candidates = [
        os.path.join(time_dir, f"scan {scan_id}", "new"),
        os.path.join(time_dir, f"scan{scan_id}", "new"),
        os.path.join(time_dir, f"scan {scan_id}", "New"),
        os.path.join(time_dir, f"scan{scan_id}", "New"),
    ]
    for c in candidates:
        if os.path.isdir(c):
            return c

    best = None
    for root, dirs, files in os.walk(time_dir):
        if os.path.basename(root).lower() != "new":
            continue
        parent = os.path.basename(os.path.dirname(root)).lower()
        if "scan" in parent and str(scan_id) in parent:
            if best is None or len(root) < len(best):
                best = root
    return best


def detect_file_col(fieldnames):
    if not fieldnames:
        return None
    candidates = ["file", "filename", "image", "img", "name", "path"]
    low = [c.lower() for c in fieldnames]
    for k in candidates:
        if k in low:
            return fieldnames[low.index(k)]
    return None


def read_csv_value_map(csv_path: str, value_col_priority="CSJ_mm"):
    if not os.path.exists(csv_path):
        return {}

    with open(csv_path, "r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            return {}
        file_col = detect_file_col(reader.fieldnames)
        if file_col is None:
            return {}

        if value_col_priority in reader.fieldnames:
            val_col = value_col_priority
        else:
            val_col = None
            for cand in ["CSJ_mm", "csj_mm", "CSJ", "csj", "value", "Value"]:
                if cand in reader.fieldnames:
                    val_col = cand
                    break
            if val_col is None:
                for c in reader.fieldnames:
                    if c != file_col:
                        val_col = c
                        break
            if val_col is None:
                return {}

        out = {}
        for row in reader:
            fn = row.get(file_col, "")
            v = row.get(val_col, "")
            if not fn or v == "":
                continue
            try:
                out[os.path.basename(fn).lower()] = float(v)
            except Exception:
                continue
        return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_root", type=str, required=True)
    ap.add_argument("--csj_filename", type=str, default="csj.cvs")
    ap.add_argument("--csj_value_col", type=str, default="CSJ_mm")
    ap.add_argument("--scan_id", type=int, default=1)
    ap.add_argument("--use_splits", type=str, default="all", choices=["all", "train", "valid", "test"])
    ap.add_argument("--aggregate", type=str, default="per_case", choices=["per_case", "all_images"],
                    help="per_case=每个 eye/case 等权；all_images=所有图像直接统计")
    ap.add_argument("--out_dir", type=str, default="./csj_stat_out")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    case_values = {t: [] for t in TIME_MIN_LIST}
    image_values = {t: [] for t in TIME_MIN_LIST}

    split_roots = detect_split_roots(args.dataset_root, args.use_splits)

    for split_root in split_roots:
        for subj_dir in list_subject_dirs(split_root):
            for eye in ["OD", "OS"]:
                eye_dir = os.path.join(subj_dir, eye)
                if not os.path.isdir(eye_dir):
                    continue

                per_case_tmp = {t: [] for t in TIME_MIN_LIST}

                for tmin in TIME_MIN_LIST:
                    time_dir = find_time_dir(eye_dir, tmin)
                    if time_dir is None:
                        continue

                    csv_path = os.path.join(time_dir, args.csj_filename)
                    csj_map = read_csv_value_map(csv_path, value_col_priority=args.csj_value_col)
                    if not csj_map:
                        continue

                    img_dir = find_scan_new_dir(time_dir, scan_id=args.scan_id)
                    if img_dir is None:
                        continue

                    for fn in os.listdir(img_dir):
                        if not fn.lower().endswith(IMG_EXTS):
                            continue
                        key = fn.lower()
                        if key in csj_map:
                            v = csj_map[key]
                            per_case_tmp[tmin].append(v)
                            image_values[tmin].append(v)

                for tmin in TIME_MIN_LIST:
                    if len(per_case_tmp[tmin]) > 0:
                        case_values[tmin].append(float(np.mean(per_case_tmp[tmin])))

    rows = []
    for tmin in TIME_MIN_LIST:
        if args.aggregate == "per_case":
            arr = np.array(case_values[tmin], dtype=np.float64)
            n = len(case_values[tmin])
        else:
            arr = np.array(image_values[tmin], dtype=np.float64)
            n = len(image_values[tmin])

        median = float(np.median(arr)) if n > 0 else float("nan")
        rows.append([tmin, n, median])

    csv_out = os.path.join(args.out_dir, "csj_summary_median.csv")
    with open(csv_out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["time_min", "n", "median_CSJ_mm"])
        w.writerows(rows)

    x = [r[0] for r in rows]
    y = [r[2] for r in rows]

    plt.figure(figsize=(9, 5))
    plt.plot(x, y, marker="o")
    plt.xlabel("Time after lens wear (minutes)")
    plt.ylabel("Sagittal height (CSJ_mm) ")
    plt.title(f"CSJ median over time ")

    # 时间比例真实：x 轴直接用分钟数
    xt = [0, 30, 60, 120, 180, 240]
    xl = ["0", "30m", "1h", "2h", "3h", "4h"]
    plt.xticks(xt, xl)

    plt.grid(True, linestyle="--", linewidth=0.5)
    plt.tight_layout()

    fig_out = os.path.join(args.out_dir, "csj_over_time_Gen.png")
    plt.savefig(fig_out, dpi=200)

    print("Saved:", csv_out)
    print("Saved:", fig_out)
    print("Counts by time:", {t: (len(case_values[t]) if args.aggregate=="per_case" else len(image_values[t])) for t in TIME_MIN_LIST})


if __name__ == "__main__":
    main()


# python cs_zhexiantu.py --dataset_root "Y:\quguangbuzheng_all_use" --out_dir "D:\deeplearning\WTW\results\zhexiantu" --aggregate all_images
