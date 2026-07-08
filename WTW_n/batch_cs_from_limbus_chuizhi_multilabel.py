# -*- coding: utf-8 -*-
import os
import csv
import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage import io, exposure, morphology, filters
from skimage.filters.rank import entropy
from skimage.morphology import disk
from scipy import ndimage
from scipy.ndimage import gaussian_filter1d

# ===================== 配置（请修改） =====================
IMAGE_FOLDER = r"Y:\quguangbuzheng_all_use\SCL_QGBZ_1049_Zheng Shifan_all\OS\2. initial\scan 1\predict"# 必填：分割图/原图所在文件夹（需与 CSV 的 file 列对应）
CSV_PATH     = r"Y:\quguangbuzheng_all_use\SCL_QGBZ_1049_Zheng Shifan_all\OS\2. initial\wtw_results.csv" # 必填：batch_limbus_wtw.py 导出的 CSV（含 xL_px,yL_px,xR_px,yR_px 或毫米坐标）
OUT_CSV      = r"Y:\quguangbuzheng_all_use\SCL_QGBZ_1049_Zheng Shifan_all\OS\2. initial\csj.cvs"  # 必填：输出 CS 结果 CSV
SAVE_PREVIEW = True
PREVIEW_DIR  = r"Y:\quguangbuzheng_all_use\SCL_QGBZ_1049_Zheng Shifan_all\OS\2. initial\csj_preview"  # 可留空；留空则默认 IMAGE_FOLDER/cs_preview

#2. initial
#3. 30mins
#4. 1h
#5. 2h
#6. 4h




# 成像物理尺寸（与你前面脚本一致）
SCAN_WIDTH_MM = 16.0
SCAN_DEPTH_MM = 11.989

# ===== 多类别分割标签（戴巩膜镜时）=====
TEAR_LABEL  = 102
TISSUE_LABEL = 153
# =====================================

# =========================================================


# ------------------- 前表面提取（原图/分割图均可） -------------------
def remove_HA(image):
    img = image.astype(np.float64)
    row_mean = img.mean(axis=1, keepdims=True)
    return img - row_mean

def gradient_vertical(I, scale=0.4, option=1):
    from skimage.transform import rescale, resize
    small = rescale(I, scale, anti_aliasing=True, preserve_range=True).astype(np.float64)
    se = np.array([[1.0], [-1.0]]) if option == 1 else np.array([[-1.0],[1.0]])
    grad_small = ndimage.convolve(small, se, mode='nearest')
    grad_small = np.abs(grad_small)
    grad = resize(grad_small, I.shape, anti_aliasing=True, preserve_range=True)
    m = grad.max() if grad.max() > 0 else 1.0
    return grad / m

def create_mask_from_gradient(grad):
    g8 = (grad * 255).astype(np.uint8)
    ent = entropy(g8, disk(15))
    if ent.max() > 0:
        ent = (ent / ent.max() * 255).astype(np.uint8)
    thr = filters.threshold_otsu(ent.astype(np.float64)/255.0) - 0.05
    thr = max(thr, 0.0)
    bw = (ent.astype(np.float64)/255.0) > thr
    bw = morphology.closing(bw, morphology.disk(3))
    bw = ndimage.binary_fill_holes(bw)
    bw = morphology.remove_small_objects(bw, min_size=10000)
    bw = morphology.opening(bw, morphology.disk(6))
    bw = morphology.erosion(bw, morphology.disk(3))
    return bw.astype(bool)

def coarse_edge_from_mask(mask):
    h, w = mask.shape
    edges = morphology.binary_dilation(mask) ^ mask
    y = np.full(w, np.nan)
    for x in range(w):
        col = np.where(edges[:, x])[0]
        if col.size > 0:
            y[x] = col.min()
    valid = ~np.isnan(y)
    if valid.sum() < 10:
        # 兜底：使用 mask 第一非零行
        y = np.full(w, np.nan)
        for x in range(w):
            col = np.where(mask[:, x])[0]
            if col.size > 0:
                y[x] = col.min()
        valid = ~np.isnan(y)
        if valid.sum() < 10:
            raise RuntimeError("Coarse edge failed.")
    return np.interp(np.arange(w), np.where(valid)[0], y[valid])

def linear_trans(x):
    xa, xb = 0.6, 0.8
    ya, yb = 0.1, 0.8
    y = np.zeros_like(x)
    m1 = x <= xa
    y[m1] = ya/xa * x[m1]
    m2 = (x > xa) & (x <= xb)
    y[m2] = (yb-ya)/(xb-xa)*x[m2] + (ya*xb - xa*yb)/(xb-xa)
    m3 = x > xb
    y[m3] = (yb-1)/(xb-1)*x[m3] + (xb-yb)/(xb-1)
    return y

def dp_path(grad_win, w_dia=1.0):
    I = grad_win.astype(np.float64)
    I = I / (I.max() if I.max()>0 else 1.0)
    I = linear_trans(I)

    H, W = I.shape
    l = np.zeros((H, W), dtype=np.float64)
    paths = np.zeros((H, W), dtype=np.int32)

    for j in range(1, W):
        # 第一行
        s1 = (2 - I[0,j-1] - I[0,j]) + l[0,j-1]
        s2 = (2 - I[1,j-1] - I[0,j]) * w_dia + l[1,j-1]
        if s1 < s2:
            l[0,j], paths[0,j] = s1, 0
        else:
            l[0,j], paths[0,j] = s2, 1
        # 中间
        for i in range(1, H-1):
            cands = [
                (2 - I[i-1,j-1] - I[i,j]) * w_dia + l[i-1,j-1],
                (2 - I[i  ,j-1] - I[i,j])            + l[i  ,j-1],
                (2 - I[i+1,j-1] - I[i,j]) * w_dia + l[i+1,j-1],
            ]
            p = int(np.argmin(cands))
            l[i,j] = cands[p]
            paths[i,j] = i + p - 1
        # 最后一行
        s1 = (2 - I[H-2,j-1] - I[H-1,j]) * w_dia + l[H-2,j-1]
        s2 = (2 - I[H-1,j-1] - I[H-1,j]) + l[H-1,j-1]
        if s1 < s2:
            l[H-1,j], paths[H-1,j] = s1, H-2
        else:
            l[H-1,j], paths[H-1,j] = s2, H-1

    path = np.zeros(W, dtype=np.int32)
    path[-1] = int(np.argmin(l[:, -1]))
    for j in range(W-1, 0, -1):
        path[j-1] = paths[path[j], j]
    return path

def refine_surface(grad, coarse_edge, win_frac=0.1):
    h, w = grad.shape
    win_h = max(int(h * win_frac), 20)
    grad_win = np.zeros((win_h, w), dtype=np.float64)
    for x in range(w):
        y0 = int(round(coarse_edge[x]))
        y0 = np.clip(y0, 0, h - win_h)
        grad_win[:, x] = grad[y0:y0+win_h, x]
    path = dp_path(grad_win, w_dia=1.0)
    y = np.zeros(w, dtype=np.float64)
    for x in range(w):
        y0 = int(round(coarse_edge[x]))
        y0 = np.clip(y0, 0, h - win_h)
        y[x] = y0 + path[x]
    return gaussian_filter1d(y, sigma=w/40)

def extract_surface_from_seg(seg):
    """分割图提取前表面（像素 y_surf）。

    - 二值/单类：每列最上面的非背景像素
    - 戴巩膜镜多类：优先取 TEAR_LABEL→TISSUE_LABEL 交界（避免把镜片当表面）
    """
    seg = seg.astype(np.uint16, copy=False)
    H, W = seg.shape
    vals = np.unique(seg)

    def _interp(y):
        ok = np.isfinite(y)
        if ok.sum() < 3:
            return None
        return np.interp(np.arange(W), np.where(ok)[0], y[ok])

    if len(vals) < 64 and (TEAR_LABEL in vals) and (TISSUE_LABEL in vals):
        y = np.full(W, np.nan, dtype=np.float64)
        for x in range(W):
            col = seg[:, x]
            m = (col[:-1] == TEAR_LABEL) & (col[1:] == TISSUE_LABEL)
            idx = np.where(m)[0]
            if idx.size:
                y[x] = float(idx[0])
        y2 = _interp(y)
        if y2 is not None:
            return y2

    vals2, cnts = np.unique(seg, return_counts=True)
    bg = int(vals2[np.argmax(cnts)])
    y = np.full(W, np.nan, dtype=np.float64)
    for x in range(W):
        col = seg[:, x]
        idx = np.where(col != bg)[0]
        if idx.size:
            y[x] = float(idx[0])
    return _interp(y)

def anterior_surface(image_path):
    """从原始 OCT 或分割图提取前表面（像素 y_surf）"""
    img = io.imread(image_path)
    if img.ndim == 3:
        img = img[..., 0]
    img = img.astype(np.float64)

    # 分割图（离散值很少）→ 直接走列顶点法
    unique_vals = np.unique(img)
    if len(unique_vals) < 32:
        y = extract_surface_from_seg(img.astype(np.uint8))
        if y is not None:
            return y, img.shape

    # 原始灰度图 → 梯度 + DP
    img_corr = remove_HA(img)
    img_corr = img_corr - img_corr.min()
    m = img_corr.max()
    img_corr = img_corr / (m if m > 0 else 1.0)
    img_corr = exposure.equalize_adapthist(img_corr, clip_limit=0.01)

    grad = gradient_vertical(img_corr, scale=0.4, option=1)
    mask = create_mask_from_gradient(grad)
    coarse = coarse_edge_from_mask(mask)
    y = refine_surface(grad, coarse, win_frac=0.1)
    return y, img.shape


# ------------------- 矢高（apex→chord 的垂直距离） -------------------
def apex_to_chord_distance_and_foot(xL_mm, zL_mm, xR_mm, zR_mm):
    """
    以 apex=(0,0) 为原点；给定 chord 两端点（毫米坐标），
    计算 apex 到 chord 的垂直距离（最短距离）以及垂足(foot)坐标（毫米）。
    """
    # 线的一般式：A x + B z + C = 0，通过两点 (xL,zL), (xR,zR)
    A = zR_mm - zL_mm
    B = xL_mm - xR_mm
    C = xR_mm * zL_mm - xL_mm * zR_mm
    denom = np.hypot(A, B)
    if denom == 0:
        return np.nan, np.nan, np.nan
    # 距离（apex 在原点）= |C|/sqrt(A^2 + B^2)
    dist = abs(C) / denom
    # 垂足（原点到直线的垂足），对一般式有解析解（参考点为原点）
    # 垂足 (xf, zf) = (-A*C/(A^2+B^2), -B*C/(A^2+B^2))
    xf = -A * C / (A*A + B*B)
    zf = -B * C / (A*A + B*B)
    return float(dist), float(xf), float(zf)

def sagittal_depth_from_profile(x_mm, z_mm, chord_len_mm):
    """
    旧口径：以 apex 为弦中点 & 原点，给定弦长 C 时的 vertical sag（仅用于 CS10）
    """
    C = chord_len_mm
    xL = -C/2.0
    xR =  C/2.0
    if xL < x_mm.min() or xR > x_mm.max():
        return np.nan
    zL, z0, zR = np.interp([xL, 0.0, xR], x_mm, z_mm)
    slope = (zR - zL) / (xR - xL)
    z_line_0 = zL + slope * (0.0 - xL)
    #return float(z_line_0 - z0)
    return float(abs(z_line_0 - z0))  # ← 永远非负


# ------------------- 稳健 float 解析 -------------------
def f2n_any(v):
    if v is None:
        return None
    if isinstance(v, str):
        v = v.strip()
        if v == "" or v.lower() in ("none", "nan", "na", "null"):
            return None
    try:
        x = float(v)
        if np.isnan(x) or np.isinf(x):
            return None
        return x
    except:
        return None


# ------------------- 批处理主流程 -------------------
def process_batch(image_folder, csv_path, out_csv,
                  save_preview=True, preview_dir=None,
                  chord_10mm=True):

    if not preview_dir and save_preview:
        preview_dir = os.path.join(image_folder, "cs_preview")
    if save_preview:
        os.makedirs(preview_dir, exist_ok=True)
    os.makedirs(os.path.dirname(out_csv) if os.path.dirname(out_csv) else ".", exist_ok=True)

    # 读取 limbus CSV
    rows = []
    with open(csv_path, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)

    results = []
    for i, r in enumerate(rows):
        fname = r.get("file") or r.get("filename") or r.get("name")
        if not fname:
            print(f"[{i+1}] 跳过：CSV 缺少 file 列")
            continue
        img_path = os.path.join(image_folder, fname)
        if not os.path.exists(img_path):
            print(f"[{i+1}] 跳过：找不到图像 {img_path}")
            continue

        # 读取 Limbus 像素坐标
        xL_px = f2n_any(r.get("xL_px")); yL_px = f2n_any(r.get("yL_px"))
        xR_px = f2n_any(r.get("xR_px")); yR_px = f2n_any(r.get("yR_px"))
        # 读取毫米坐标（兜底用）
        xL_mm_csv = f2n_any(r.get("xL_mm")); zL_mm_csv = f2n_any(r.get("zL_mm"))
        xR_mm_csv = f2n_any(r.get("xR_mm")); zR_mm_csv = f2n_any(r.get("zR_mm"))

        # 前表面与 apex
        try:
            y_surf, (H, W) = anterior_surface(img_path)
        except Exception as e:
            print(f"[{i+1}] 跳过：前表面提取失败（{fname}）：{e}")
            continue

        realw = SCAN_WIDTH_MM / (W - 1)
        realh = SCAN_DEPTH_MM / H
        xs = np.arange(W)

        # apex：中间 1/3 最浅点（像素）
        mid = (xs > W/3) & (xs < 2*W/3)
        idx_apex = mid.nonzero()[0][np.argmin(y_surf[mid])]
        x_apex = float(idx_apex); y_apex = float(y_surf[idx_apex])

        # 若像素 Limbus 坐标缺失，尝试毫米→像素回推
        if (xL_px is None or yL_px is None or xR_px is None or yR_px is None):
            if (xL_mm_csv is not None and zL_mm_csv is not None and
                xR_mm_csv is not None and zR_mm_csv is not None):
                xL_px = x_apex + xL_mm_csv/realw
                yL_px = y_apex - zL_mm_csv/realh
                xR_px = x_apex + xR_mm_csv/realw
                yR_px = y_apex - zR_mm_csv/realh
            else:
                print(f"[{i+1}] 跳过：缺 Limbus 像素坐标且 CSV 无毫米坐标（{fname}）")
                continue

        # 再次校验
        if any(v is None for v in (xL_px, yL_px, xR_px, yR_px)):
            print(f"[{i+1}] 跳过：Limbus 像素坐标无效（{fname}）")
            continue

        # 组装弦端点（毫米，apex 为原点）
        xL_mm = (xL_px - x_apex) * realw
        zL_mm = (y_apex - yL_px) * realh
        xR_mm = (xR_px - x_apex) * realw
        zR_mm = (y_apex - yR_px) * realh

        # 整条前表面（毫米，apex 为原点）——用于 CS10
        x_mm = (xs - x_apex) * realw
        z_mm = (y_apex - y_surf) * realh

        # --- 目标：CSJ = apex 到 chord 的垂直距离 ---
        CSJ_mm, foot_x_mm, foot_z_mm = apex_to_chord_distance_and_foot(xL_mm, zL_mm, xR_mm, zR_mm)

        # CS10（可选，旧口径）
        CS10 = sagittal_depth_from_profile(x_mm, z_mm, chord_len_mm=10.0) if chord_10mm else None

        rec = {
            "file": fname,
            "apex_x_px": x_apex, "apex_y_px": y_apex,
            "xL_px": xL_px, "yL_px": yL_px,
            "xR_px": xR_px, "yR_px": yR_px,
            "xL_mm": xL_mm, "zL_mm": zL_mm,
            "xR_mm": xR_mm, "zR_mm": zR_mm,
            "WTW_mm": float(np.hypot(xR_mm - xL_mm, zR_mm - zL_mm)),
            "CSJ_mm": CSJ_mm,          # 你要的严格定义：apex→chord 垂直距离
            "CS10_mm": CS10
        }
        results.append(rec)

        # 预览图
        if save_preview:
            try:
                img = io.imread(img_path)
                if img.ndim == 3: img = img[...,0]
                plt.figure(figsize=(6,4))
                plt.imshow(img, cmap='gray')
                plt.plot(xs, y_surf, 'r-', lw=1.0, label='Anterior surface')
                plt.scatter([x_apex], [y_apex], c='y', s=30, label='Apex')
                plt.scatter([xL_px, xR_px], [yL_px, yR_px], c='c', s=30, label='Limbus')

                # 弦（毫米直线→像素绘制）
                slope = 0.0 if np.isclose(xL_mm, xR_mm) else (zR_mm - zL_mm) / (xR_mm - xL_mm)
                xm = np.linspace(min(xL_mm, xR_mm)-2, max(xL_mm, xR_mm)+2, 200)
                zm = zL_mm + slope * (xm - xL_mm)
                xp = x_apex + xm / realw
                yp = y_apex - zm / realh
                plt.plot(xp, yp, 'g--', lw=0.8, label='Chord')

                # 垂足（像素）& 垂线段（apex -> foot）
                foot_x_px = x_apex + foot_x_mm / realw
                foot_z_px = y_apex - foot_z_mm / realh
                plt.plot([x_apex, foot_x_px], [y_apex, foot_z_px], 'w-', lw=1.4,
                         label=f'CSJ={CSJ_mm:.3f} mm')

                Hh, Ww = img.shape[:2]
                plt.xlim(0, Ww-1); plt.ylim(Hh-1, 0)
                plt.legend(fontsize=7, loc='lower right')
                ttl = f"{fname}\nWTW={rec['WTW_mm']:.3f}  CSJ={CSJ_mm:.3f}"
                if CS10 is None or np.isnan(CS10):
                    ttl += "  CS10=NA"
                else:
                    ttl += f"  CS10={CS10:.3f}"
                plt.title(ttl)
                plt.tight_layout()
                out_dir = PREVIEW_DIR if PREVIEW_DIR else os.path.join(IMAGE_FOLDER, "cs_preview")
                os.makedirs(out_dir, exist_ok=True)
                out_png = os.path.join(out_dir, os.path.splitext(fname)[0] + "_cs.png")
                plt.savefig(out_png, dpi=160)
                plt.close()
            except Exception as e:
                print(f"[{i+1}] 预览图失败（{fname}）：{e}")

        msg = f"[{i+1}/{len(rows)}] {fname}  WTW={rec['WTW_mm']:.3f}  CSJ={CSJ_mm:.3f}"
        if CS10 is None or np.isnan(CS10):
            msg += "  CS10=NA"
        else:
            msg += f"  CS10={CS10:.3f}"
        print(msg)

    # 写 CSV
    fieldnames = ["file","apex_x_px","apex_y_px",
                  "xL_px","yL_px","xR_px","yR_px",
                  "xL_mm","zL_mm","xR_mm","zR_mm",
                  "WTW_mm","CSJ_mm","CS10_mm"]
    with open(out_csv, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in results:
            w.writerow({
                k: (None if v is None or (isinstance(v,float) and (np.isnan(v) or np.isinf(v)))
                    else (f"{v:.6f}" if isinstance(v,(float,np.floating)) else v))
                for k,v in r.items()
            })
    print(f"\nCS 结果已写入：{out_csv}")
    if save_preview:
        print(f"预览图目录：{PREVIEW_DIR if PREVIEW_DIR else os.path.join(IMAGE_FOLDER, 'cs_preview')}")

if __name__ == "__main__":
    if not IMAGE_FOLDER or not CSV_PATH or not OUT_CSV:
        print("请先编辑脚本顶部配置：IMAGE_FOLDER / CSV_PATH / OUT_CSV")
    else:
        process_batch(IMAGE_FOLDER, CSV_PATH, OUT_CSV, SAVE_PREVIEW, PREVIEW_DIR, chord_10mm=True)

