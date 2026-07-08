import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from numpy.polynomial import Polynomial

# ========== 配置 ==========
folder = r"D:\deeplearning\WTW\seg_ak"

SCAN_WIDTH_MM = 16.0      # Casia2 radial 扫描长度 (mm)
SCAN_DEPTH_MM = 11.989    # 在组织中的成像深度 (mm)，先按已校正处理
MAX_SHOW = 3              # 随机画几帧看效果
# =========================


def list_bscan_files(folder):
    files = [f for f in os.listdir(folder) if f.lower().endswith('.png')]
    if not files:
        raise RuntimeError("没有找到 tif 文件")

    shapes = []
    cache = {}
    for f in files:
        p = os.path.join(folder, f)
        img = cv2.imread(p, -1)
        if img is None:
            continue
        cache[f] = img
        shapes.append(img.shape)
    if not shapes:
        raise RuntimeError("tif 读取失败")

    unique, counts = np.unique(shapes, axis=0, return_counts=True)
    main_shape = unique[np.argmax(counts)]
    bscans = [f for f in files if cache[f].shape == tuple(main_shape)]
    print(f"识别到主尺寸 B-scan: {main_shape}, 共 {len(bscans)} 帧")
    return sorted(bscans), main_shape


def ensure_zx(img):
    # 保证 (z, x) = (深度, 横向)
    z, x = img.shape
    if z < x:
        return img
    else:
        return img.T


def find_anterior_surface(img):
    img = img.astype(np.float32)
    z_max, x_max = img.shape
    surf = np.zeros(x_max, dtype=np.float32)
    for j in range(x_max):
        col = img[:, j]
        col = cv2.GaussianBlur(col.reshape(-1, 1), (1, 5), 0).ravel()
        grad = np.diff(col)
        search_end = max(int(z_max * 0.35), 20)
        z0 = np.argmax(grad[:search_end])
        surf[j] = z0
    return surf


def second_moment_curve(img, z_min=0, z_max=None):
    if z_max is None or z_max > img.shape[0]:
        z_max = img.shape[0]
    img = img.astype(np.float32)
    z_idx = np.arange(z_min, z_max, dtype=np.float32)
    m2 = np.zeros(img.shape[1], dtype=np.float32)
    for j in range(img.shape[1]):
        col = img[z_min:z_max, j]
        w = col + 1e-6
        s = np.sum(w)
        z_mean = np.sum(z_idx * w) / s
        m2[j] = np.sum(((z_idx - z_mean)**2) * w) / s
    return m2


def smooth(y, k=31):
    k = min(k, len(y)//2*2-1)
    if k < 3:
        return y
    ker = np.ones(k) / k
    return np.convolve(y, ker, mode='same')


def rough_limbus_by_band(x_mm, dm, band_min, band_max):
    """在给定 x(mm) 区间里，取 |dm| 最大的点作为粗略 limbus"""
    idx = np.where((x_mm >= band_min) & (x_mm <= band_max))[0]
    if idx.size == 0:
        return None
    sub = np.abs(dm[idx])
    j = idx[np.argmax(sub)]
    return int(j)


def fit_local_limbus(x_mm, z_mm, j0, cornea_span=2.5, sclera_span=2.5):
    """以 j0 为中心局部拟合角膜/巩膜，求交点"""
    if j0 is None:
        return None, None

    x0 = x_mm[j0]
    x_loc = x_mm - x0

    cor_mask = (x_loc <= 0) & (x_loc >= -cornea_span)
    scl_mask = (x_loc >= 0) & (x_loc <= sclera_span)

    c_idx = np.where(cor_mask)[0]
    s_idx = np.where(scl_mask)[0]
    if c_idx.size < 15 or s_idx.size < 15:
        return None, None

    p_cor = Polynomial.fit(x_loc[c_idx], z_mm[c_idx], 4).convert()
    p_scl = Polynomial.fit(x_loc[s_idx], z_mm[s_idx], 2).convert()

    p_diff = p_cor - p_scl
    roots = p_diff.roots()
    roots = roots[np.isreal(roots)].real
    if roots.size == 0:
        return None, None

    x_l_loc = roots[np.argmin(np.abs(roots))]
    if abs(x_l_loc) > max(cornea_span, sclera_span):
        return None, None

    x_l = x_l_loc + x0
    z_l = p_cor(x_l_loc)
    return x_l, z_l


def process_one_scan(folder, max_show=3):
    bfiles, (Nz, Nx) = list_bscan_files(folder)

    realw = SCAN_WIDTH_MM / (Nx - 1)
    realh = SCAN_DEPTH_MM / Nz
    print(f"横向 {realw*1000:.2f} µm/px, 纵向 {realh*1000:.2f} µm/px")

    raw_results = []
    shown = 0

    for i, fname in enumerate(bfiles):
        img = cv2.imread(os.path.join(folder, fname), -1)
        if img is None:
            continue
        img = ensure_zx(img)
        Nz, Nx = img.shape

        x_pix = np.arange(Nx, dtype=np.float32)
        x_mm = (x_pix - Nx/2.0) * realw

        z_surf = find_anterior_surface(img)
        z_mm = z_surf * realh

        m2 = second_moment_curve(img)
        m2_s = smooth(m2, k=31)
        dm = np.gradient(m2_s)

        # 粗略 limbus：用解剖范围限制
        jL_rough = rough_limbus_by_band(x_mm, dm, -7.0, -4.0)
        jR_rough = rough_limbus_by_band(x_mm, dm,  4.0,  7.0)

        xL, zL = fit_local_limbus(x_mm, z_mm, jL_rough)
        xR, zR = fit_local_limbus(x_mm, z_mm, jR_rough)

        if xL is not None and xR is not None:
            WTW = np.sqrt((xR - xL)**2 + (zR - zL)**2)
            raw_results.append((i, fname, xL, zL, xR, zR, WTW))

        # 可视化
        if shown < max_show:
            shown += 1
            plt.figure(figsize=(6,4))
            plt.imshow(img, cmap='gray', origin='upper')
            plt.plot(x_pix, z_surf, '.', ms=0.5, label='Anterior surface')
            if jL_rough is not None:
                plt.axvline(jL_rough, ls='--', lw=0.6, label='L rough')
            if jR_rough is not None:
                plt.axvline(jR_rough, ls='--', lw=0.6, label='R rough')
            if xL is not None:
                plt.scatter(xL/realw + Nx/2, zL/realh, s=25, label='L limbus')
            if xR is not None:
                plt.scatter(xR/realw + Nx/2, zR/realh, s=25, label='R limbus')
            plt.gca().invert_yaxis()
            plt.title(f"B-scan {i}: {fname}")
            plt.legend(fontsize=6, loc='lower right')
            plt.tight_layout()
            plt.show()

    print("—— 原始 WTW (mm) ——")
    for r in raw_results:
        i, fname, xL, zL, xR, zR, WTW = r
        print(f"{i:2d} {fname:20s}  {WTW:.3f} mm")

    # ===== 质量控制 / 过滤 =====
    filtered = []
    for (i, fname, xL, zL, xR, zR, WTW) in raw_results:
        # 1) 合理范围
        if not (10.0 <= WTW <= 12.5):
            continue
        # 2) 左右对称性（只看横向）
        if abs(abs(xL) - abs(xR)) > 1.5:  # 两侧中心距差太大，踢掉
            continue
        filtered.append((i, fname, xL, zL, xR, zR, WTW))

    print("—— 通过过滤的 WTW (mm) ——")
    for r in filtered:
        i, fname, xL, zL, xR, zR, WTW = r
        print(f"{i:2d} {fname:20s}  {WTW:.3f} mm")

    if not filtered:
        print("没有通过过滤的帧，请检查参数。")
        return None

    # 最终 WTW：推荐用中位数，抗 outlier
    wtws = np.array([r[6] for r in filtered])
    final_median = float(np.median(wtws))
    final_mean = float(np.mean(wtws))

    print("\n========== 最终结果 ==========")
    print(f"该次扫描最终 topographic WTW (median) = {final_median:.3f} mm")
    print(f"(参考) 平均值 WTW (mean)           = {final_mean:.3f} mm")

    return {
        "raw": raw_results,
        "filtered": filtered,
        "WTW_median": final_median,
        "WTW_mean": final_mean,
    }


if __name__ == "__main__":
    res = process_one_scan(folder, MAX_SHOW)
