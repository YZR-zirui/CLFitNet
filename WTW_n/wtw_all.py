import os
import re
import csv
import numpy as np
import cv2
import matplotlib.pyplot as plt
from numpy.polynomial import Polynomial
from scipy.signal import savgol_filter, find_peaks

# ===================== 配置 =====================
FOLDER = r"Y:\SC\sc\data\QGBZ\ScCL_QGBZ_1066_Wang Qian\OS\Label"                # 所有时刻分割图都在这个 Label 文件夹
OUT_ROOT = r"Y:\SC\sc\data\QGBZ\ScCL_QGBZ_1066_Wang Qian\OS\WTW_result"         # 输出根目录：会自动生成各时刻子文件夹
SAVE_PREVIEW = True                      # 是否保存可视化叠加图

# 成像物理尺寸（与设备一致）
SCAN_WIDTH_MM = 16.0
SCAN_DEPTH_MM = 11.989

# ===== 多类别分割标签（戴巩膜镜时）=====
TEAR_LABEL   = 102   # 泪液/液体层
TISSUE_LABEL = 153   # 角膜/眼表组织
# =====================================

# 解剖搜索带（mm）：在这些 x(mm) 区间内找曲率峰作为粗定位
SEARCH_LEFT_MM  = (-7.0, -4.0)
SEARCH_RIGHT_MM = ( 4.0,  7.0)

# 拟合窗口（mm）：以粗略 limbus 为中心，内侧=角膜，外侧=巩膜
CORNEA_SPAN_MM = 2.5
SCLERA_SPAN_MM = 2.5

# 平滑 / 峰距（mm）
SMOOTH_WIN_MM    = 0.6
PEAK_MIN_DIST_MM = 0.6
# =================================================


def read_seg(path: str):
    """
    读分割图（优先 cv2.imread；失败时用 imdecode 兜底，兼容中文路径问题）。
    """
    seg = cv2.imread(path, -1)
    if seg is None:
        data = np.fromfile(path, dtype=np.uint8)
        if data.size == 0:
            raise RuntimeError(f"读取失败(文件不存在或不可读): {path}")
        seg = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)

    if seg is None:
        raise RuntimeError(f"读取失败(无法解码为图像): {path}")

    if seg.ndim == 3:
        seg = cv2.cvtColor(seg, cv2.COLOR_BGR2GRAY)
    return seg


def spacing_mm(seg, scan_w_mm=16.0, scan_d_mm=11.989):
    H, W = seg.shape
    return scan_w_mm / (W - 1), scan_d_mm / H


def _interp_nan_1d(y):
    x = np.arange(len(y))
    ok = np.isfinite(y)
    if ok.sum() < 3:
        return y
    return np.interp(x, x[ok], y[ok])


def _extract_boundary(seg, upper_vals, lower_vals):
    """找每列中 upper->lower 的最浅交界（返回 upper 像素的 y）。"""
    H, W = seg.shape
    y = np.full(W, np.nan, dtype=np.float32)
    upper_vals = np.array(list(upper_vals), dtype=seg.dtype)
    lower_vals = np.array(list(lower_vals), dtype=seg.dtype)
    for j in range(W):
        col = seg[:, j]
        m = np.isin(col[:-1], upper_vals) & np.isin(col[1:], lower_vals)
        idx = np.where(m)[0]
        if idx.size:
            y[j] = float(idx[0])
    return y


def extract_surface(seg, bg=None):
    """
    提取前表面（像素 y 坐标）：
    - 多类分割：优先取 TEAR_LABEL -> TISSUE_LABEL 的交界
    - 兜底：BG -> TISSUE_LABEL
    - 单类/二值：每列最上面的非背景像素
    """
    seg = seg.astype(np.uint16, copy=False)
    H, W = seg.shape
    vals = np.unique(seg)

    # 1) 多类别：TEAR->TISSUE
    if len(vals) < 64 and (TEAR_LABEL in vals) and (TISSUE_LABEL in vals):
        y = _extract_boundary(seg, upper_vals=[TEAR_LABEL], lower_vals=[TISSUE_LABEL])

        # 兜底：部分列没 TEAR，就试 BG->TISSUE
        if np.isfinite(y).sum() < 0.2 * W:
            y2 = _extract_boundary(seg, upper_vals=[0], lower_vals=[TISSUE_LABEL])
            miss = ~np.isfinite(y)
            y[miss] = y2[miss]

        if np.isfinite(y).sum() >= 3:
            return _interp_nan_1d(y)

    # 2) 单类/二值：列顶点法
    if bg is None:
        vals2, cnts = np.unique(seg, return_counts=True)
        bg = int(vals2[np.argmax(cnts)])
    y = np.full(W, np.nan, dtype=np.float32)
    for j in range(W):
        col = seg[:, j]
        idx = np.where(col != bg)[0]
        if idx.size:
            y[j] = float(idx[0])
    return _interp_nan_1d(y)


def smooth_surface_mm(z_mm, realw, win_mm=0.6):
    """在 mm 轴上做 SG 平滑，窗长根据 realw 自适应。"""
    valid = ~np.isnan(z_mm)
    if valid.sum() < 21:
        return z_mm
    win = int(max(7, 2 * int((win_mm / realw) // 2) + 1))  # 奇数
    poly = 3 if win >= 7 else 2
    z = z_mm.copy()
    z[valid] = savgol_filter(
        z[valid],
        window_length=min(win, valid.sum() // 2 * 2 + 1),
        polyorder=poly
    )
    return z


def curvature_mm(x_mm, z_mm):
    """曲率 κ = |z''| / (1+z'^2)^(3/2)。"""
    dzdx = np.gradient(z_mm, x_mm)
    d2zdx = np.gradient(dzdx, x_mm)
    kappa = np.abs(d2zdx) / np.maximum((1.0 + dzdx ** 2) ** 1.5, 1e-6)
    kappa[~np.isfinite(kappa)] = 0.0
    return kappa


def rough_from_kappa(x_mm, kappa, band, min_distance_mm=0.6):
    """在 band 内找曲率峰的索引作为粗定位。"""
    idx = np.where((x_mm >= band[0]) & (x_mm <= band[1]))[0]
    if idx.size == 0:
        return None
    k_sub = kappa[idx]
    dx = np.mean(np.diff(x_mm))
    dist = max(1, int(min_distance_mm / dx))
    peaks, _ = find_peaks(k_sub, distance=dist)
    j_local = peaks[np.argmax(k_sub[peaks])] if peaks.size else int(np.argmax(k_sub))
    return int(idx[j_local])


def local_fit_intersection(x_mm, z_mm, j0, side, cor_span=2.5, scl_span=2.5):
    """
    以 j0 为中心分段拟合：角膜4阶、巩膜2阶，求交点 = Limbus。
    """
    if j0 is None or np.isnan(z_mm[j0]):
        return None, None, None, None
    x0 = x_mm[j0]
    xloc = x_mm - x0

    if side == 'left':
        c_mask = (xloc >= 0) & (xloc <= cor_span)
        s_mask = (xloc <= 0) & (xloc >= -scl_span)
    else:
        c_mask = (xloc <= 0) & (xloc >= -cor_span)
        s_mask = (xloc >= 0) & (xloc <= scl_span)

    valid = ~np.isnan(z_mm)
    c_idx = np.where(c_mask & valid)[0]
    s_idx = np.where(s_mask & valid)[0]
    if c_idx.size < 12 or s_idx.size < 12:
        return None, None, None, None

    p_cor = Polynomial.fit(xloc[c_idx], z_mm[c_idx], 4).convert()
    p_scl = Polynomial.fit(xloc[s_idx], z_mm[s_idx], 2).convert()
    roots = (p_cor - p_scl).roots()
    roots = roots[np.isreal(roots)].real
    if roots.size == 0:
        return None, None, p_cor, p_scl

    u = roots[np.argmin(np.abs(roots))]
    if abs(u) > max(cor_span, scl_span):
        return None, None, p_cor, p_scl
    return x0 + u, float(p_cor(u)), p_cor, p_scl


def fmt(x):
    if x is None:
        return None
    if isinstance(x, float) and (np.isnan(x) or np.isinf(x)):
        return None
    if isinstance(x, (float, np.floating)):
        return f"{x:.6f}"
    return x


# ========= 关键修改：加入 I / Initial 时刻 =========
def parse_timepoint(fname: str):
    """
    从文件名中提取时刻：
      - I / Initial  (例如 JLY_OD_I_img055.png)
      - 30min
      - 1H / 2H / 4H (例如 WHH_OD_4H_img036.png)
    """
    base = os.path.splitext(os.path.basename(fname))[0]

    # 支持 _I_ 或 _Initial_（不区分大小写）
    m = re.search(r'_(i|initial)_', base, flags=re.IGNORECASE)
    if m:
        return "I"

    # 支持 30min/1h/2h/4h
    m = re.search(r'_(30min|30mins|1h|2h|4h)_', base, flags=re.IGNORECASE)
    if not m:
        return None
    t = m.group(1).lower()
    if t in ("30min", "30mins"):
        return "30min"
    if t == "1h":
        return "1H"
    if t == "2h":
        return "2H"
    if t == "4h":
        return "4H"
    return None


def timepoint_to_outdir(tp: str) -> str:
    """
    输出子文件夹命名规则：
      I -> 2.Initial
      其他 -> 自身名字
    你如果想把 I 输出到 "I" 文件夹，把这里改成 return tp 即可。
    """
    if tp == "I":
        return "2.Initial"
    return tp
# ================================================


def process_files(files, folder, out_csv, save_preview=True, preview_dir=None):
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    if save_preview and preview_dir:
        os.makedirs(preview_dir, exist_ok=True)

    all_rows = []
    for idx, fname in enumerate(files):
        path = os.path.join(folder, fname)

        try:
            seg = read_seg(path)
        except Exception as e:
            print(f"[跳过] {fname}: {e}")
            continue

        H, W = seg.shape
        realw, realh = spacing_mm(seg, SCAN_WIDTH_MM, SCAN_DEPTH_MM)

        z_surf_pix = extract_surface(seg, bg=None)
        x_pix = np.arange(W, dtype=np.float32)
        x_mm = (x_pix - W / 2.0) * realw
        z_mm = z_surf_pix * realh
        z_mm = smooth_surface_mm(z_mm, realw, SMOOTH_WIN_MM)

        kappa = curvature_mm(x_mm, z_mm)
        jL = rough_from_kappa(x_mm, kappa, SEARCH_LEFT_MM, PEAK_MIN_DIST_MM)
        jR = rough_from_kappa(x_mm, kappa, SEARCH_RIGHT_MM, PEAK_MIN_DIST_MM)

        xLmm, zLmm, pcl, psl = local_fit_intersection(
            x_mm, z_mm, jL, 'left', CORNEA_SPAN_MM, SCLERA_SPAN_MM
        )
        xRmm, zRmm, pcr, psr = local_fit_intersection(
            x_mm, z_mm, jR, 'right', CORNEA_SPAN_MM, SCLERA_SPAN_MM
        )

        xLpx = xLmm / realw + W / 2.0 if xLmm is not None else None
        yLpx = zLmm / realh if zLmm is not None else None
        xRpx = xRmm / realw + W / 2.0 if xRmm is not None else None
        yRpx = zRmm / realh if zRmm is not None else None

        WTW = None
        if xLmm is not None and xRmm is not None and zLmm is not None and zRmm is not None:
            WTW = float(np.hypot(xRmm - xLmm, zRmm - zLmm))

        all_rows.append({
            "index": idx,
            "file": fname,
            "xL_px": xLpx, "yL_px": yLpx,
            "xR_px": xRpx, "yR_px": yRpx,
            "xL_mm": xLmm, "zL_mm": zLmm,
            "xR_mm": xRmm, "zR_mm": zRmm,
            "WTW_mm": WTW
        })

        if save_preview and preview_dir:
            plt.figure(figsize=(6, 4))
            plt.imshow(seg, cmap='gray', origin='upper')
            plt.plot(x_pix, z_surf_pix, '.', ms=0.6, label='Anterior surface')

            if jL is not None:
                plt.axvline(jL, ls='--', lw=0.8, label='L rough')
            if jR is not None:
                plt.axvline(jR, ls='--', lw=0.8, label='R rough')

            u = np.linspace(-CORNEA_SPAN_MM, SCLERA_SPAN_MM, 400)
            if pcl is not None and psl is not None and jL is not None:
                xplot_pix = (u + x_mm[jL]) / realw + W / 2.0
                plt.plot(xplot_pix, pcl(u) / realh, 'g-', lw=1.2, label='Cornea fit (L)')
                plt.plot(xplot_pix, psl(u) / realh, 'b-', lw=1.2, label='Sclera fit (L)')
            if pcr is not None and psr is not None and jR is not None:
                xplot_pix = (u + x_mm[jR]) / realw + W / 2.0
                plt.plot(xplot_pix, pcr(u) / realh, 'g--', lw=1.2, label='Cornea fit (R)')
                plt.plot(xplot_pix, psr(u) / realh, 'b--', lw=1.2, label='Sclera fit (R)')

            if xLpx is not None:
                plt.scatter([xLpx], [yLpx], s=35, c='r', label='L limbus', zorder=5)
            if xRpx is not None:
                plt.scatter([xRpx], [yRpx], s=35, c='y', label='R limbus', zorder=5)

            plt.title(f"{fname}  WTW={WTW:.3f} mm" if WTW else f"{fname}  (WTW=NA)")
            plt.legend(fontsize=7, loc='lower right')
            plt.tight_layout()
            out_png = os.path.join(preview_dir, os.path.splitext(fname)[0] + "_limbus.png")
            plt.savefig(out_png, dpi=160)
            plt.close()

        print(f"[{idx + 1}/{len(files)}] {fname}  WTW={WTW:.3f} mm" if WTW else
              f"[{idx + 1}/{len(files)}] {fname}  WTW=NA")

    with open(out_csv, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.writer(f)
        w.writerow([
            "index", "file",
            "xL_px", "yL_px", "xR_px", "yR_px",
            "xL_mm", "zL_mm", "xR_mm", "zR_mm", "WTW_mm"
        ])
        for r in all_rows:
            w.writerow([
                r["index"], r["file"],
                fmt(r["xL_px"]), fmt(r["yL_px"]),
                fmt(r["xR_px"]), fmt(r["yR_px"]),
                fmt(r["xL_mm"]), fmt(r["zL_mm"]),
                fmt(r["xR_mm"]), fmt(r["zR_mm"]),
                fmt(r["WTW_mm"])
            ])

    print(f"\n结果已写入：{out_csv}")
    if save_preview and preview_dir:
        print(f"预览图保存在：{preview_dir}")


def process_folder_grouped_by_time(folder, out_root, save_preview=True):
    exts = ('.png', '.tif', '.tiff', '.jpg', '.bmp')
    files = [f for f in os.listdir(folder) if f.lower().endswith(exts)]
    files.sort()
    if not files:
        raise RuntimeError("文件夹内未找到分割图像")

    # 加入 I
    groups = {"I": [], "30min": [], "1H": [], "2H": [], "4H": [], "unknown": []}

    for f in files:
        tp = parse_timepoint(f)
        if tp is None:
            groups["unknown"].append(f)
        else:
            groups[tp].append(f)

    os.makedirs(out_root, exist_ok=True)

    # 输出顺序你也可以改
    order = ["I", "30min", "1H", "2H", "4H", "unknown"]
    for tp in order:
        flist = groups.get(tp, [])
        if not flist:
            continue

        sub = timepoint_to_outdir(tp)
        out_dir = os.path.join(out_root, sub)
        out_csv = os.path.join(out_dir, "wtw_results.csv")
        preview_dir = os.path.join(out_dir, "wtw_preview")

        print(f"\n===== 时刻 {tp}（输出到 {sub}）：{len(flist)} 张 =====")
        process_files(
            flist, folder, out_csv,
            save_preview=save_preview,
            preview_dir=(preview_dir if save_preview else None)
        )

    if groups["unknown"]:
        print("\n[提示] 有些文件没识别到时刻（放进 unknown 组，仅提示不影响其它结果）：")
        for f in groups["unknown"][:30]:
            print("  ", f)
        if len(groups["unknown"]) > 30:
            print("  ...")


if __name__ == "__main__":
    process_folder_grouped_by_time(FOLDER, OUT_ROOT, SAVE_PREVIEW)
