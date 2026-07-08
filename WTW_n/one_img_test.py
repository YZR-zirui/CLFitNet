import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from numpy.polynomial import Polynomial

# ======== 配置 ========
img_path = r"D:\deeplearning\WTW\TSJ_OD_B_img005.tif"
SCAN_WIDTH_MM = 16.0
SCAN_DEPTH_MM = 11.989

band_mm = 0.30        # 前表面下多深的带来算M2
cornea_span_mm = 2.5  # limbus内侧拟合范围
sclera_span_mm = 2.5  # limbus外侧拟合范围
margin_px = 60        # 两端安全距离, 避开logo/黑边
# =====================


def ensure_zx(img):
    z, x = img.shape
    # Casia B-scan 多数是 (depth, lateral) 即 z < x
    return img if z < x else img.T


def find_anterior_surface(img):
    img = img.astype(np.float32)
    z_max, x_max = img.shape
    surf = np.zeros(x_max, dtype=np.float32)
    for j in range(x_max):
        col = cv2.GaussianBlur(img[:, j].reshape(-1,1), (1,5), 0).ravel()
        grad = np.diff(col)
        search_end = max(int(z_max * 0.35), 20)
        z0 = np.argmax(grad[:search_end])
        surf[j] = z0
    return surf


def second_moment_near_surface(img, z_surf, realh, band_mm):
    """
    只在前表面下 band_mm 深度范围算二阶中心矩，
    模仿论文“信号分布变化”，避免深部噪声/图标。
    """
    img = img.astype(np.float32)
    z_max, x_max = img.shape
    band_px = max(int(band_mm / realh), 5)

    m2 = np.zeros(x_max, dtype=np.float32)
    for j in range(x_max):
        z0 = int(z_surf[j])
        if z0 < 0 or z0 >= z_max - 5:
            continue
        z1 = min(z_max, z0 + band_px)
        if z1 - z0 < 5:
            continue

        col = img[z0:z1, j]
        z_idx = np.arange(z0, z1, dtype=np.float32)
        w = col + 1e-6
        s = np.sum(w)
        z_mean = np.sum(z_idx * w) / s
        m2[j] = np.sum(((z_idx - z_mean) ** 2) * w) / s

    return m2


def smooth(y, k=21):
    k = min(k, len(y)//2*2-1)
    if k < 3:
        return y
    ker = np.ones(k)/k
    return np.convolve(y, ker, mode='same')


def rough_limbus_by_band_safe(x_mm, dm, band_min, band_max, margin_px, Nx):
    """
    在指定x(mm)范围内, 且避开左右margin, 找|dm|最大点。
    """
    idx = np.where((x_mm >= band_min) & (x_mm <= band_max))[0]
    idx = idx[(idx > margin_px) & (idx < Nx - margin_px)]
    if idx.size == 0:
        return None
    sub = np.abs(dm[idx])
    return int(idx[np.argmax(sub)])


def fit_limbus_side(x_mm, z_mm, j0, side,
                    cornea_span_mm=2.5, sclera_span_mm=2.5):
    """
    最关键的一步：在局部坐标下分角膜/巩膜两段拟合并求交点。
    side = 'left' 或 'right'
    定义:
      - 内侧: 朝 x=0 的方向
      - 外侧: 朝远离 x=0 的方向
    """
    if j0 is None:
        return None, None, None, None

    x0 = x_mm[j0]
    x_loc = x_mm - x0

    if side == 'left':  # x0 < 0
        # 内侧(角膜): 从 limbus 向右(朝0) cornea_span_mm
        cor_mask = (x_mm >= x0) & (x_mm <= x0 + cornea_span_mm)
        # 外侧(巩膜): 从 limbus 向左 sclera_span_mm
        scl_mask = (x_mm <= x0) & (x_mm >= x0 - sclera_span_mm)
    else:  # 'right', x0 > 0
        # 内侧(角膜): 从 limbus 向左(朝0) cornea_span_mm
        cor_mask = (x_mm <= x0) & (x_mm >= x0 - cornea_span_mm)
        # 外侧(巩膜): 从 limbus 向右 sclera_span_mm
        scl_mask = (x_mm >= x0) & (x_mm <= x0 + sclera_span_mm)

    c_idx = np.where(cor_mask)[0]
    s_idx = np.where(scl_mask)[0]
    if c_idx.size < 15 or s_idx.size < 15:
        return None, None, None, None

    # 用局部坐标拟合, 数值更稳
    p_cor = Polynomial.fit(x_loc[c_idx], z_mm[c_idx], 4).convert()
    p_scl = Polynomial.fit(x_loc[s_idx], z_mm[s_idx], 2).convert()

    p_diff = p_cor - p_scl
    roots = p_diff.roots()
    roots = roots[np.isreal(roots)].real
    if roots.size == 0:
        return None, None, p_cor, p_scl

    # 选离0最近的根 => 过渡点
    u = roots[np.argmin(np.abs(roots))]
    if abs(u) > max(cornea_span_mm, sclera_span_mm):
        return None, None, p_cor, p_scl

    x_l = x0 + u
    z_l = p_cor(u)
    return x_l, z_l, p_cor, p_scl


# ========== 跑这一帧 ==========
img = cv2.imread(img_path, -1)
if img is None:
    raise RuntimeError("图像读取失败")
img = ensure_zx(img)
Nz, Nx = img.shape

realw = SCAN_WIDTH_MM / (Nx - 1)
realh = SCAN_DEPTH_MM / Nz

x_pix = np.arange(Nx, dtype=np.float32)
x_mm = (x_pix - Nx/2.0) * realw

z_surf = find_anterior_surface(img)
z_mm = z_surf * realh

m2 = second_moment_near_surface(img, z_surf, realh, band_mm)
m2_s = smooth(m2, k=21)
dm = np.gradient(m2_s)

# 左侧粗略 limbus: -7 ~ -4 mm
jL = rough_limbus_by_band_safe(x_mm, dm, -7.0, -4.0, margin_px, Nx)
# 右侧粗略 limbus: 4 ~ 7 mm
jR = rough_limbus_by_band_safe(x_mm, dm,  4.0,  7.0, margin_px, Nx)

xL, zL, p_cor_L, p_scl_L = fit_limbus_side(
    x_mm, z_mm, jL, 'left', cornea_span_mm, sclera_span_mm
)
xR, zR, p_cor_R, p_scl_R = fit_limbus_side(
    x_mm, z_mm, jR, 'right', cornea_span_mm, sclera_span_mm
)

# ========== 画图 ==========
plt.figure(figsize=(6,4))
plt.imshow(img, cmap='gray', origin='upper')
plt.plot(x_pix, z_surf, '.', ms=0.4, label='Anterior surface')

# 左右粗略位置
if jL is not None:
    plt.axvline(jL, ls='--', lw=0.6, label='L rough')
if jR is not None:
    plt.axvline(jR, ls='--', lw=0.6, label='R rough')

u = np.linspace(-cornea_span_mm, sclera_span_mm, 400)

# 左侧拟合曲线
if p_cor_L is not None and p_scl_L is not None and jL is not None:
    x0 = x_mm[jL]
    x_plot = (u + x0)/realw + Nx/2.0
    plt.plot(x_plot, p_cor_L(u)/realh, 'g-', lw=1.0, label='Cornea fit (L)')
    plt.plot(x_plot, p_scl_L(u)/realh, 'b-', lw=1.0, label='Sclera fit (L)')

# 右侧拟合曲线
if p_cor_R is not None and p_scl_R is not None and jR is not None:
    x0 = x_mm[jR]
    x_plot = (u + x0)/realw + Nx/2.0
    plt.plot(x_plot, p_cor_R(u)/realh, 'g--', lw=1.0, label='Cornea fit (R)')
    plt.plot(x_plot, p_scl_R(u)/realh, 'b--', lw=1.0, label='Sclera fit (R)')

# Limbus 点
if xL is not None:
    plt.scatter(xL/realw + Nx/2.0, zL/realh, s=40, c='r', label='L limbus')
if xR is not None:
    plt.scatter(xR/realw + Nx/2.0, zR/realh, s=40, c='y', label='R limbus')

plt.gca().invert_yaxis()
plt.legend(fontsize=6, loc='lower right')
plt.tight_layout()
plt.show()

if xL is not None and xR is not None:
    WTW = np.sqrt((xR - xL)**2 + (zR - zL)**2)
    print(f"WTW = {WTW:.3f} mm")
else:
    print("某一侧 limbus 未成功拟合。")
