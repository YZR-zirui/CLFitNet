# -*- coding: utf-8 -*-
# batch_limbus_wtw_qc.py
import os, csv, numpy as np, cv2, matplotlib.pyplot as plt
from numpy.polynomial import Polynomial
from scipy.signal import savgol_filter, find_peaks

# ===== 配置 =====
FOLDER    = r"D:\deeplearning\WTW\seg_ak"   # 分割图文件夹
OUT_CSV   = r"D:\deeplearning\WTW\results\WTW\limbus_wtw_select_results.csv"
SAVE_PREV = True
PREV_DIR  = r"D:\deeplearning\WTW\results\WTW\preview1"

# 物理视野（mm）
SCAN_W_MM = 16.0
SCAN_H_MM = 11.989   # 或者用你的折射率校正值

# 搜索带（mm）
SEARCH_LEFT  = (-7.0, -4.0)
SEARCH_RIGHT = ( 4.0,  7.0)
# 拟合窗口（mm）
CORNEA_SPAN  = 2.5
SCLERA_SPAN  = 2.5
# 平滑窗口（mm）与峰距（mm）
SMOOTH_WIN   = 0.6
PEAK_DIST_MM = 0.6

# WTW 合理范围（用于强条件 QC）
WTW_MIN, WTW_MAX = 8.5, 14.8
# =================

def read_seg(path):
    seg = cv2.imread(path, -1)
    if seg is None: raise RuntimeError(f"读取失败: {path}")
    if seg.ndim == 3: seg = cv2.cvtColor(seg, cv2.COLOR_BGR2GRAY)
    return seg

def spacing_mm(seg):
    H,W = seg.shape
    return SCAN_W_MM/(W-1), SCAN_H_MM/H

def extract_surface(seg):
    H,W = seg.shape
    vals, cnts = np.unique(seg, return_counts=True)
    bg = int(vals[np.argmax(cnts)])
    z = np.full(W, np.nan, np.float32)
    for j in range(W):
        col = seg[:, j]
        idx = np.where(col != bg)[0]
        if idx.size: z[j] = idx[0]
    return z

def smooth_mm(z_mm, realw, win_mm=0.6):
    valid = ~np.isnan(z_mm)
    if valid.sum() < 21: return z_mm
    win = int(max(7, 2*int((win_mm/realw)//2)+1))
    poly = 3 if win >= 7 else 2
    z = z_mm.copy()
    z[valid] = savgol_filter(z[valid], window_length=min(win, valid.sum()//2*2+1), polyorder=poly)
    return z

def curvature(x_mm, z_mm):
    dz = np.gradient(z_mm, x_mm)
    d2 = np.gradient(dz, x_mm)
    k = np.abs(d2) / np.maximum((1+dz**2)**1.5, 1e-6)
    k[~np.isfinite(k)] = 0.0
    return k

def rough_from_kappa(x_mm, kappa, band, min_dist_mm):
    idx = np.where((x_mm>=band[0])&(x_mm<=band[1]))[0]
    if idx.size==0: return None
    k_sub = kappa[idx]
    dx = np.mean(np.diff(x_mm))
    dist = max(1, int(min_dist_mm/dx))
    peaks, _ = find_peaks(k_sub, distance=dist)
    j_local = peaks[np.argmax(k_sub[peaks])] if peaks.size else int(np.argmax(k_sub))
    return int(idx[j_local])

def local_fit_intersection(x_mm, z_mm, j0, side, cor_span, scl_span):
    if j0 is None or np.isnan(z_mm[j0]): return None,None,None,None
    x0 = x_mm[j0]; xloc = x_mm - x0
    if side=='left':
        cmask = (xloc>=0)&(xloc<=cor_span)
        smask = (xloc<=0)&(xloc>=-scl_span)
    else:
        cmask = (xloc<=0)&(xloc>=-cor_span)
        smask = (xloc>=0)&(xloc<=scl_span)
    valid = ~np.isnan(z_mm)
    ci = np.where(cmask & valid)[0]; si = np.where(smask & valid)[0]
    if ci.size<12 or si.size<12: return None,None,None,None
    pcor = Polynomial.fit(xloc[ci], z_mm[ci], 4).convert()
    pscl = Polynomial.fit(xloc[si], z_mm[si], 2).convert()
    roots = (pcor-pscl).roots(); roots = roots[np.isreal(roots)].real
    if roots.size==0: return None,None,pcor,pscl
    u = roots[np.argmin(np.abs(roots))]
    if abs(u)>max(cor_span,scl_span): return None,None,pcor,pscl
    return x0+u, float(pcor(u)), pcor, pscl

def fmt(x):
    if x is None: return None
    if isinstance(x,(float,np.floating)):
        if np.isnan(x) or np.isinf(x): return None
        return f"{x:.6f}"
    return x

def process_folder(folder, out_csv, save_preview=True, preview_dir=None):
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    if save_preview and preview_dir: os.makedirs(preview_dir, exist_ok=True)
    files = [f for f in os.listdir(folder) if f.lower().endswith(('.png','.tif','.tiff','.jpg','.bmp'))]
    files.sort()
    if not files: raise RuntimeError("未找到分割图像")

    rows=[]
    for k,fname in enumerate(files,1):
        path = os.path.join(folder, fname)
        seg  = read_seg(path)
        H,W  = seg.shape
        realw, realh = spacing_mm(seg)

        # 曲面
        z_surf = extract_surface(seg)
        x_pix  = np.arange(W, dtype=np.float32)
        x_mm   = (x_pix - W/2.0)*realw              # 图像基准毫米坐标：x=0在画面中线
        z_mm   = z_surf * realh                     # z向下为正（WTW阶段用）
        z_mm   = smooth_mm(z_mm, realw, SMOOTH_WIN)

        # limbus 粗定位 & 精定位
        kappa = curvature(x_mm, z_mm)
        jL = rough_from_kappa(x_mm, kappa, SEARCH_LEFT,  PEAK_DIST_MM)
        jR = rough_from_kappa(x_mm, kappa, SEARCH_RIGHT, PEAK_DIST_MM)
        xLmm,zLmm,pl,sl = local_fit_intersection(x_mm, z_mm, jL, 'left',  CORNEA_SPAN, SCLERA_SPAN)
        xRmm,zRmm,pr,sr = local_fit_intersection(x_mm, z_mm, jR, 'right', CORNEA_SPAN, SCLERA_SPAN)

        # 像素坐标（用于后续 CS 脚本）
        xLpx = xLmm/realw + W/2.0 if xLmm is not None else None
        yLpx = zLmm/realh         if zLmm is not None else None
        xRpx = xRmm/realw + W/2.0 if xRmm is not None else None
        yRpx = zRmm/realh         if zRmm is not None else None

        # WTW 与 QC
        reason = []
        if xLmm is None or xRmm is None:
            WTW = None; reason.append("limbus_not_found")
        else:
            WTW = float(np.hypot(xRmm-xLmm, zRmm-zLmm))
            if not (WTW_MIN <= WTW <= WTW_MAX):
                reason.append("WTW_out_of_range")

        rows.append({"index":k-1, "file":fname,
                     "xL_px":xLpx,"yL_px":yLpx,"xR_px":xRpx,"yR_px":yRpx,
                     "xL_mm":xLmm,"zL_mm":zLmm,"xR_mm":xRmm,"zR_mm":zRmm,
                     "WTW_mm":WTW, "qc_reason":"|".join(reason) if reason else ""})

        # 预览
        if save_preview and preview_dir:
            plt.figure(figsize=(6,4))
            plt.imshow(seg, cmap='gray', origin='upper')
            plt.plot(x_pix, z_surf, '.', ms=0.6, label='Anterior surface')
            if jL is not None: plt.axvline(jL, ls='--', lw=0.8, label='L rough')
            if jR is not None: plt.axvline(jR, ls='--', lw=0.8, label='R rough')
            if xLpx is not None: plt.scatter([xLpx],[yLpx], s=35, c='r', label='L limbus')
            if xRpx is not None: plt.scatter([xRpx],[yRpx], s=35, c='y', label='R limbus')
            plt.gca().invert_yaxis()
            ttl = f"{fname}  WTW={WTW:.3f} mm" if WTW else f"{fname}  (WTW=NA)"
            if reason: ttl += f"  [{','.join(reason)}]"
            plt.title(ttl); plt.legend(fontsize=7, loc='lower right'); plt.tight_layout()
            out_png = os.path.join(preview_dir, os.path.splitext(fname)[0]+"_limbus.png")
            plt.savefig(out_png, dpi=160); plt.close()

        msg = f"[{k}/{len(files)}] {fname}  WTW={WTW:.3f} mm" if WTW else f"[{k}/{len(files)}] {fname}  WTW=NA"
        if reason: msg += f"  -> {reason}"
        print(msg)

    with open(out_csv,'w',newline='',encoding='utf-8-sig') as f:
        w = csv.writer(f)
        w.writerow(["index","file","xL_px","yL_px","xR_px","yR_px","xL_mm","zL_mm","xR_mm","zR_mm","WTW_mm","qc_reason"])
        for r in rows:
            w.writerow([r["index"], r["file"],
                        fmt(r["xL_px"]), fmt(r["yL_px"]),
                        fmt(r["xR_px"]), fmt(r["yR_px"]),
                        fmt(r["xL_mm"]), fmt(r["zL_mm"]),
                        fmt(r["xR_mm"]), fmt(r["zR_mm"]),
                        fmt(r["WTW_mm"]), r["qc_reason"]])

if __name__=="__main__":
    process_folder(FOLDER, OUT_CSV, SAVE_PREV, PREV_DIR)
