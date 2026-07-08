

# -*- coding: utf-8 -*-
"""
aggregate_per_eye_with_ci_standalone.py

读取逐帧 CSV（含 WTW/CSJ/CS10 与 qc_reason），
在“强条件 + 可选离群值”筛选后，按眼别聚合：
  median / IQR / n / 95% CI（中位数的 bootstrap CI）
并输出每眼质量等级（基于 IQR）与被剔除帧清单（可选）。

依赖：numpy, pandas
"""

import os, re
import numpy as np
import pandas as pd
from collections import defaultdict

# ====================== 配置：把你的路径改在这里 ======================
CS_CSV   = r"D:\deeplearning\WTW\results\cs_new_process\cv.cvs"              # 逐帧结果
OUT_CSV  = r"D:\deeplearning\WTW\results\T_OD_eye_summary.csv" # 每眼汇总
REJ_CSV  = r"D:\deeplearning\WTW\results\rejected_frames.csv"

# ---------------- 参数（可按需调整） ----------------
# 强条件（不满足将被排除）
WTW_RANGE = (8.5, 14.8)        # 合理 WTW 范围（mm）
REQUIRE_CSJ_FOOT_T = True      # 若 qc_reason 含 foot_outside_chord，则剔除该帧的 CSJ
REQUIRE_CS10_FOV   = True      # 若 qc_reason 含 FOV_insufficient_for_CS10，则剔除该帧的 CS10

# 可选：离群值（相对各眼的“帧中位数”）
USE_OUTLIER_FILTER = True
OUTLIER_THR = {
    "WTW_mm": 0.80,            # |WTW - median| > 0.80 mm → 剔除
    "CSJ_mm": 0.60,            # |CSJ - median| > 0.60 mm → 剔除
    "CS10_mm": 0.50            # |CS10 - median| > 0.50 mm → 剔除
}

# IQR 质量等级（方便报告）
IQR_GRADES = {
    "WTW":  (0.30, 0.60),      # ≤0.30 优；(0.30,0.60] 可；>0.60 需复核
    "CSJ":  (0.20, 0.50),
    "CS10": (0.20, 0.50)
}

# 95% CI（中位数）bootstrap 参数
N_BOOT = 2000
ALPHA  = 0.05
RNG_SEED = 2025
# ---------------------------------------------------

SUBJ_REGEX = re.compile(r"^([A-Za-z0-9]+)_(OD|OS)_", re.IGNORECASE)

def parse_subject_eye(fname: str):
    m = SUBJ_REGEX.search(os.path.basename(fname))
    if not m: return (os.path.splitext(fname)[0], "UNK")
    return (m.group(1), m.group(2).upper())

def f2f(x):
    try:
        if x is None: return np.nan
        s = str(x).strip().lower()
        if s in ("", "na", "nan", "none", "null"): return np.nan
        return float(s)
    except:
        return np.nan

def iqr(arr):
    arr = np.asarray(arr, float)
    return float(np.nanpercentile(arr, 75) - np.nanpercentile(arr, 25))

def grade_from_iqr(metric: str, v: float):
    if not np.isfinite(v): return ""
    t1, t2 = IQR_GRADES[metric]
    if v <= t1: return "Good"
    if v <= t2: return "Fair"
    return "Review"

def ci_median_bootstrap(values, n_boot=N_BOOT, alpha=ALPHA, seed=RNG_SEED):
    v = np.array(values, float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    meds = np.empty(n_boot, dtype=float)
    n = v.size
    for i in range(n_boot):
        sample = rng.choice(v, size=n, replace=True)
        meds[i] = np.median(sample)
    lo = np.percentile(meds, 100*alpha/2)
    hi = np.percentile(meds, 100*(1-alpha/2))
    return (float(lo), float(hi))

def apply_strong_filters(df: pd.DataFrame) -> pd.DataFrame:
    """返回：保留原列 + 三个布尔列，标明各指标是否有效。"""
    df = df.copy()
    for col in ["WTW_mm","CSJ_mm","CS10_mm"]:
        if col not in df.columns:
            df[col] = np.nan
    if "qc_reason" not in df.columns:
        df["qc_reason"] = ""

    # 三个有效性标记
    ok_wtw  = np.isfinite(df["WTW_mm"].map(f2f))
    ok_csj  = np.isfinite(df["CSJ_mm"].map(f2f))
    ok_cs10 = np.isfinite(df["CS10_mm"].map(f2f))

    # WTW 合理范围
    wtw = df["WTW_mm"].map(f2f).astype(float)
    ok_wtw &= (wtw >= WTW_RANGE[0]) & (wtw <= WTW_RANGE[1])

    # CSJ 垂足在线段内（依赖上游 qc_reason）
    if REQUIRE_CSJ_FOOT_T:
        ok_csj &= ~df["qc_reason"].str.contains("foot_outside_chord", case=False, na=False)

    # CS10 视野覆盖（依赖上游 qc_reason）
    if REQUIRE_CS10_FOV:
        ok_cs10 &= ~df["qc_reason"].str.contains("FOV_insufficient_for_CS10", case=False, na=False)

    df["ok_WTW"]  = ok_wtw
    df["ok_CSJ"]  = ok_csj
    df["ok_CS10"] = ok_cs10
    return df

def apply_outlier_filter_per_eye(df: pd.DataFrame) -> pd.DataFrame:
    """按眼做二次离群剔除；仅对已经通过强条件的帧起作用。"""
    df = df.copy()
    df["ol_WTW"] = False
    df["ol_CSJ"] = False
    df["ol_CS10"] = False

    groups = df.groupby(["subject","eye"], dropna=False)
    for (_, _), g in groups:
        # WTW
        g_ok = g[g["ok_WTW"]]
        if len(g_ok) > 0:
            med = np.nanmedian(g_ok["WTW_mm"])
            mask = np.abs(g["WTW_mm"] - med) > OUTLIER_THR["WTW_mm"]
            df.loc[g.index, "ol_WTW"] = mask.values & g["ok_WTW"].values
        # CSJ
        g_ok = g[g["ok_CSJ"]]
        if len(g_ok) > 0:
            med = np.nanmedian(g_ok["CSJ_mm"])
            mask = np.abs(g["CSJ_mm"] - med) > OUTLIER_THR["CSJ_mm"]
            df.loc[g.index, "ol_CSJ"] = mask.values & g["ok_CSJ"].values
        # CS10
        g_ok = g[g["ok_CS10"]]
        if len(g_ok) > 0:
            med = np.nanmedian(g_ok["CS10_mm"])
            mask = np.abs(g["CS10_mm"] - med) > OUTLIER_THR["CS10_mm"]
            df.loc[g.index, "ol_CS10"] = mask.values & g["ok_CS10"].values

    return df

def summarize_per_eye(df: pd.DataFrame) -> pd.DataFrame:
    """输出每眼：median / IQR / n / 95% CI + 质量等级。"""
    recs = []
    for (subj, eye), g in df.groupby(["subject","eye"], dropna=False):
        for metric, ok_col, ol_col in [
            ("WTW",  "ok_WTW",  "ol_WTW"),
            ("CSJ",  "ok_CSJ",  "ol_CSJ"),
            ("CS10", "ok_CS10", "ol_CS10"),
        ]:
            col = metric + "_mm"
            if col not in g.columns: continue
            mask = g[ok_col] & ~g[ol_col]
            vals = g.loc[mask, col].astype(float).values
            n = int(np.sum(np.isfinite(vals)))
            med = float(np.nanmedian(vals)) if n>0 else np.nan
            IQR = float(iqr(vals)) if n>=4 else np.nan
            lo, hi = ci_median_bootstrap(vals) if n>0 else (np.nan, np.nan)
            grade = grade_from_iqr(metric, IQR)
            recs.append({
                "subject": subj, "eye": eye, "metric": metric,
                "n": n,
                "median": (None if not np.isfinite(med) else round(med, 3)),
                "IQR": (None if not np.isfinite(IQR) else round(IQR, 3)),
                "CI95_lo": (None if not np.isfinite(lo) else round(lo, 3)),
                "CI95_hi": (None if not np.isfinite(hi) else round(hi, 3)),
                "grade": grade
            })
    # 展开成一行三指标
    out = defaultdict(dict)
    for r in recs:
        key = (r["subject"], r["eye"])
        out[key][r["metric"]] = r
    rows=[]
    for (subj, eye), d in out.items():
        row = {"subject": subj, "eye": eye}
        for m in ("WTW","CSJ","CS10"):
            r = d.get(m, {})
            row[f"{m}_median"]  = r.get("median")
            row[f"{m}_IQR"]     = r.get("IQR")
            row[f"{m}_n"]       = r.get("n")
            lo,hi = r.get("CI95_lo"), r.get("CI95_hi")
            row[f"{m}_CI95"]    = None if (lo is None or hi is None) else f"[{lo}, {hi}]"
            row[f"{m}_grade"]   = r.get("grade")
        rows.append(row)
    return pd.DataFrame(rows)

def main():
    if not os.path.exists(CS_CSV):
        raise FileNotFoundError(f"找不到逐帧 CSV：{CS_CSV}")

    df = pd.read_csv(CS_CSV, encoding="utf-8-sig")

    # 解析 subject/eye（若 CSV 已有则复用）
    if "subject" not in df.columns or "eye" not in df.columns:
        subjects=[]; eyes=[]
        for f in df["file"]:
            s,e = parse_subject_eye(f)
            subjects.append(s); eyes.append(e)
        df["subject"]=subjects; df["eye"]=eyes

    # 类型转换
    for col in ("WTW_mm","CSJ_mm","CS10_mm"):
        if col in df.columns: df[col] = df[col].map(f2f).astype(float)
        else: df[col] = np.nan
    if "qc_reason" not in df.columns: df["qc_reason"] = ""

    # 强条件
    df = apply_strong_filters(df)

    # 离群值（可选）
    if USE_OUTLIER_FILTER:
        df = apply_outlier_filter_per_eye(df)

    # 被剔除帧清单（可选）
    if REJ_CSV:
        rej = df[
            (~df["ok_WTW"]) | (~df["ok_CSJ"]) | (~df["ok_CS10"]) |
            (df.get("ol_WTW", False)) | (df.get("ol_CSJ", False)) | (df.get("ol_CS10", False))
        ].copy()
        os.makedirs(os.path.dirname(REJ_CSV) or ".", exist_ok=True)
        rej.to_csv(REJ_CSV, index=False, encoding="utf-8-sig")
        print("已写出被剔除帧清单：", REJ_CSV)

    # 每眼汇总
    summary = summarize_per_eye(df).sort_values(["subject","eye"]).reset_index(drop=True)
    os.makedirs(os.path.dirname(OUT_CSV) or ".", exist_ok=True)
    summary.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    print("已写出每眼汇总：", OUT_CSV)

if __name__ == "__main__":
    main()
