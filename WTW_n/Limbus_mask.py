import numpy as np
import cv2
import matplotlib.pyplot as plt
from numpy.polynomial import Polynomial
from scipy.signal import savgol_filter, find_peaks

# ===== 配置 =====
SEG_PATH = r"D:\deeplearning\WTW\seg_ak\TWB_OD_B_img002_pred3.png"  # 分割图：组织=非零，背景=0（或相反）
SCAN_WIDTH_MM = 16.0
SCAN_DEPTH_MM = 11.989

# 解剖窗口（mm）—在这个范围里找曲率峰作为粗略 L/R limbus
SEARCH_LEFT_MM  = (-7.0, -4.0)
SEARCH_RIGHT_MM = ( 4.0,  7.0)

# 拟合窗口（mm）—以粗略点为中心的局部分段拟合
CORNEA_SPAN_MM = 2.5    # 内侧（朝中心）
SCLERA_SPAN_MM = 2.5    # 外侧（远离中心）

# 平滑与峰检测
SMOOTH_WIN_MM     = 0.6   # Savitzky–Golay 平滑窗口 (mm)
PEAK_MIN_DIST_MM  = 0.6   # 同侧曲率峰最小间距 (mm)
# =================

def read_seg(path):
    seg = cv2.imread(path, -1)
    if seg is None:
        raise RuntimeError("分割图读取失败")
    if seg.ndim == 3:
        seg = cv2.cvtColor(seg, cv2.COLOR_BGR2GRAY)
    return seg

def spacing_mm(seg, scan_w_mm=16.0, scan_d_mm=11.989):
    H, W = seg.shape
    return scan_w_mm/(W-1), scan_d_mm/H

def extract_surface(seg, bg=None):
    """每列最上面的‘组织’像素为前表面（像素坐标）"""
    H, W = seg.shape
    if bg is None:
        vals, cnts = np.unique(seg, return_counts=True)
        bg = int(vals[np.argmax(cnts)])
    z = np.full(W, np.nan, dtype=np.float32)
    for j in range(W):
        col = seg[:, j]
        idx = np.where(col != bg)[0]
        if idx.size:
            z[j] = idx[0]
    return z

def smooth_surface_mm(z_mm, realw, win_mm=0.6):
    valid = ~np.isnan(z_mm)
    if valid.sum() < 21:
        return z_mm
    win = int(max(7, 2*int((win_mm/realw)//2)+1))  # 奇数
    poly = 3 if win >= 7 else 2
    z = z_mm.copy()
    z[valid] = savgol_filter(z[valid], window_length=min(win, valid.sum()//2*2+1), polyorder=poly)
    return z

def curvature_mm(x_mm, z_mm):
    dzdx  = np.gradient(z_mm, x_mm)
    d2zdx = np.gradient(dzdx, x_mm)
    kappa = np.abs(d2zdx) / np.maximum((1.0 + dzdx**2)**1.5, 1e-6)
    kappa[~np.isfinite(kappa)] = 0.0
    return kappa

def rough_from_kappa(x_mm, kappa, band, min_distance_mm=0.6):
    idx = np.where((x_mm >= band[0]) & (x_mm <= band[1]))[0]
    if idx.size == 0: return None
    k_sub = kappa[idx]
    dx = np.mean(np.diff(x_mm))
    dist = max(1, int(min_distance_mm/dx))
    peaks, _ = find_peaks(k_sub, distance=dist)
    j_local = peaks[np.argmax(k_sub[peaks])] if peaks.size else int(np.argmax(k_sub))
    return int(idx[j_local])

def local_fit_intersection(x_mm, z_mm, j0, side, cor_span=2.5, scl_span=2.5):
    """局部坐标分段拟合（角膜4阶/巩膜2阶）并求交点"""
    if j0 is None or np.isnan(z_mm[j0]):
        return None, None, None, None
    x0 = x_mm[j0]; xloc = x_mm - x0
    if side == 'left':     # 内侧→0（右），外侧→左
        c_mask = (xloc>=0) & (xloc<=cor_span)
        s_mask = (xloc<=0) & (xloc>=-scl_span)
    else:                  # right：内侧→0（左），外侧→右
        c_mask = (xloc<=0) & (xloc>=-cor_span)
        s_mask = (xloc>=0) & (xloc<=scl_span)
    valid = ~np.isnan(z_mm)
    c_idx = np.where(c_mask & valid)[0]; s_idx = np.where(s_mask & valid)[0]
    if c_idx.size < 12 or s_idx.size < 12:
        return None, None, None, None
    p_cor = Polynomial.fit(xloc[c_idx], z_mm[c_idx], 4).convert()
    p_scl = Polynomial.fit(xloc[s_idx], z_mm[s_idx], 2).convert()
    roots = (p_cor - p_scl).roots(); roots = roots[np.isreal(roots)].real
    if roots.size == 0:
        return None, None, p_cor, p_scl
    u = roots[np.argmin(np.abs(roots))]
    if abs(u) > max(cor_span, scl_span):
        return None, None, p_cor, p_scl
    return x0+u, float(p_cor(u)), p_cor, p_scl

# ========== 单帧处理 & 旧风格可视化 ==========
seg = read_seg(SEG_PATH)
H, W = seg.shape
realw, realh = spacing_mm(seg, SCAN_WIDTH_MM, SCAN_DEPTH_MM)

# 前表面（像素）→ mm 并平滑（在 mm 轴上）
z_surf_pix = extract_surface(seg, bg=None)
x_pix = np.arange(W, dtype=np.float32)
x_mm  = (x_pix - W/2.0)*realw
z_surf_mm = z_surf_pix * realh
z_surf_mm = smooth_surface_mm(z_surf_mm, realw, SMOOTH_WIN_MM)

# 曲率与粗定位
kappa = curvature_mm(x_mm, z_surf_mm)
jL = rough_from_kappa(x_mm, kappa, SEARCH_LEFT_MM,  PEAK_MIN_DIST_MM)
jR = rough_from_kappa(x_mm, kappa, SEARCH_RIGHT_MM, PEAK_MIN_DIST_MM)

# 精化 limbus（得到毫米坐标）
xL,zL,pcl,psl = local_fit_intersection(x_mm, z_surf_mm, jL, 'left',  CORNEA_SPAN_MM, SCLERA_SPAN_MM)
xR,zR,pcr,psr = local_fit_intersection(x_mm, z_surf_mm, jR, 'right', CORNEA_SPAN_MM, SCLERA_SPAN_MM)

# ---- 旧风格绘图：整张图（像素坐标），前表面 + 粗略竖线 + Limbus 点 +（可选）局部拟合 ----
plt.figure(figsize=(6,4))
plt.imshow(seg, cmap='gray', origin='upper')
# 前表面（像素）
plt.plot(x_pix, z_surf_pix, '.', ms=0.6, label='Anterior surface')

# 粗略竖线
if jL is not None: plt.axvline(jL, ls='--', lw=0.8, label='L rough')
if jR is not None: plt.axvline(jR, ls='--', lw=0.8, label='R rough')

# 画拟合曲线（把 mm → 像素）
u = np.linspace(-CORNEA_SPAN_MM, SCLERA_SPAN_MM, 400)
if pcl is not None and psl is not None and jL is not None:
    x0_pix = xL/realw + W/2.0 if xL is not None else jL
    xplot_pix = (u + x_mm[jL])/realw + W/2.0
    plt.plot(xplot_pix, pcl(u)/realh, 'g-', lw=1.2, label='Cornea fit (L)')
    plt.plot(xplot_pix, psl(u)/realh, 'b-', lw=1.2, label='Sclera fit (L)')
if pcr is not None and psr is not None and jR is not None:
    xplot_pix = (u + x_mm[jR])/realw + W/2.0
    plt.plot(xplot_pix, pcr(u)/realh, 'g--', lw=1.2, label='Cornea fit (R)')
    plt.plot(xplot_pix, psr(u)/realh, 'b--', lw=1.2, label='Sclera fit (R)')

# Limbus（像素）
if xL is not None:
    plt.scatter([xL/realw + W/2.0], [zL/realh], s=35, c='r', label='L limbus', zorder=5)
if xR is not None:
    plt.scatter([xR/realw + W/2.0], [zR/realh], s=35, c='y', label='R limbus', zorder=5)

plt.gca().invert_yaxis()
plt.title('B-scan (seg) with limbus')
plt.legend(fontsize=7, loc='lower right')
plt.tight_layout()
plt.show()

# WTW（毫米）
if xL is not None and xR is not None:
    WTW = float(np.sqrt((xR-xL)**2 + (zR-zL)**2))
    print(f"WTW = {WTW:.3f} mm ；水平距离 = {abs(xR-xL):.3f} mm")
else:
    print("某一侧 limbus 未成功定位（可调 SEARCH_*_MM / SMOOTH_WIN_MM / 拟合窗口）。")
