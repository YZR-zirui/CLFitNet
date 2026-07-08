import numpy as np
import matplotlib.pyplot as plt
from skimage import io, filters, morphology, exposure
from skimage.filters.rank import entropy
from skimage.morphology import disk
from scipy.ndimage import gaussian_filter1d
from scipy import ndimage
from numpy.polynomial import Polynomial

# ================= 标定（根据你提供的参数） =================
REALW = 16.0 / 1160.0                     # mm / px (horizontal)
REALH = 11.0 / 796.0 / 1.376              # mm / px (vertical, refractive index corrected)

# ================= 基础工具 =================

def remove_HA(image):
    """对应 RemoveHA1: 行均值校正，减少反射伪影和背景偏置。"""
    img = image.astype(np.float64)
    row_mean = img.mean(axis=1, keepdims=True)
    return img - row_mean

def gradient_image_vertical(image, scale=0.4, option=1):
    """
    对应 gradientImagecreate:
    - 下采样
    - 垂直方向卷积 [1; -1] 或 [-1; 1]
    - 上采样
    - 归一化到 [0,1]
    """
    from skimage.transform import rescale, resize

    I = image.astype(np.float64)
    small = rescale(I, scale, anti_aliasing=True, preserve_range=True)

    if option == 1:
        se = np.array([[1.0], [-1.0]])
    else:
        se = np.array([[-1.0], [1.0]])

    grad_small = ndimage.convolve(small, se, mode='nearest')
    grad_small = np.abs(grad_small)

    grad = resize(grad_small, I.shape, anti_aliasing=True, preserve_range=True)

    max_val = grad.max() if grad.max() > 0 else 1.0
    grad = grad / max_val
    return grad

def create_mask_from_gradient(grad):
    """
    仿照 CreateMask，用熵+阈值+形态学得到角膜+巩膜前表面区域。
    """
    g = (grad * 255).astype(np.uint8)

    # 局部纹理增强
    ent = entropy(g, disk(15))
    ent = (ent / ent.max() * 255).astype(np.uint8)

    # Otsu 阈值，略微下调一点
    thr = filters.threshold_otsu(ent.astype(np.float64) / 255.0) - 0.05
    thr = max(thr, 0.0)
    bw = (ent.astype(np.float64) / 255.0) > thr

    # 形态学处理
    bw = morphology.closing(bw, morphology.disk(3))
    bw = ndimage.binary_fill_holes(bw)
    bw = morphology.remove_small_objects(bw, min_size=10000)
    bw = morphology.opening(bw, morphology.disk(6))
    bw = morphology.erosion(bw, morphology.disk(3))

    return bw.astype(bool)

def get_coarse_front_edge(mask):
    """
    仿照 FrontEdge: 对每一列取 ICL_mask 边缘最上面的点作为粗边界。
    """
    h, w = mask.shape
    edges = morphology.binary_dilation(mask) ^ mask  # 边界
    y_coarse = np.full(w, np.nan)

    for x in range(w):
        col = np.where(edges[:, x])[0]
        if len(col) > 0:
            y_coarse[x] = col.min()

    # 用插值填补 NaN
    valid = ~np.isnan(y_coarse)
    if valid.sum() < 10:
        raise RuntimeError("Coarse edge detection failed: not enough valid points.")
    y_coarse = np.interp(np.arange(w), np.where(valid)[0], y_coarse[valid])
    return y_coarse

# ================= 动态规划路径（前表面精细化） =================

def linear_trans(I):
    xa, xb = 0.6, 0.8
    ya, yb = 0.1, 0.8
    x = I
    y = np.zeros_like(x)

    idx1 = x <= xa
    y[idx1] = ya / xa * x[idx1]

    idx2 = (x > xa) & (x <= xb)
    y[idx2] = (yb - ya) / (xb - xa) * x[idx2] + (ya * xb - xa * yb) / (xb - xa)

    idx3 = x > xb
    y[idx3] = (yb - 1) / (xb - 1) * x[idx3] + (xb - yb) / (xb - 1)

    return y

def weight(ga, gb):
    # 与原 MATLAB 类似，偏好强信号位置
    return 2.0 - ga - gb + 1e-4

def dynamic_programming_path(grad_win, w_dia=1.0, use_linear=True):
    """
    在给定窗口内使用动态规划找最优路径。
    grad_win: [H, W] 已经是梯度图窗口，归一化 [0,1].
    """
    I = grad_win.astype(np.float64)
    if I.max() <= 0:
        I = np.ones_like(I) * 1e-6
    else:
        I = I / I.max()

    if use_linear:
        I = linear_trans(I)

    H, W = I.shape
    l = np.zeros((H, W), dtype=np.float64)
    paths = np.zeros((H, W), dtype=np.int32)

    # 从第 2 列开始累积
    for j in range(1, W):
        # 第一行
        s1 = weight(I[0, j-1], I[0, j]) + l[0, j-1]
        s2 = weight(I[1, j-1], I[0, j]) * w_dia + l[1, j-1]
        if s1 < s2:
            l[0, j] = s1
            paths[0, j] = 0
        else:
            l[0, j] = s2
            paths[0, j] = 1

        # 中间行
        for i in range(1, H-1):
            s1 = weight(I[i-1, j-1], I[i, j]) * w_dia + l[i-1, j-1]
            s2 = weight(I[i,   j-1], I[i, j]) + l[i,   j-1]
            s3 = weight(I[i+1, j-1], I[i, j]) * w_dia + l[i+1, j-1]
            ss = [s1, s2, s3]
            p = int(np.argmin(ss))
            l[i, j] = ss[p]
            paths[i, j] = i + p - 1  # 对应上一列行号

        # 最后一行
        s1 = weight(I[H-2, j-1], I[H-1, j]) * w_dia + l[H-2, j-1]
        s2 = weight(I[H-1, j-1], I[H-1, j]) + l[H-1, j-1]
        if s1 < s2:
            l[H-1, j] = s1
            paths[H-1, j] = H-2
        else:
            l[H-1, j] = s2
            paths[H-1, j] = H-1

    # 回溯最短路径
    path = np.zeros(W, dtype=np.int32)
    path[-1] = int(np.argmin(l[:, -1]))
    for j in range(W-1, 0, -1):
        path[j-1] = paths[path[j], j]

    return path

def refine_front_surface(grad, mask, coarse_edge, win_frac=0.1):
    """
    在 coarse_edge 附近开窗口，用 DP 得到精确前表面。
    """
    h, w = grad.shape
    win_h = max(int(h * win_frac), 20)

    grad_win = np.zeros((win_h, w), dtype=np.float64)
    for x in range(w):
        y0 = int(round(coarse_edge[x]))
        y0 = np.clip(y0, 0, h - win_h)
        grad_win[:, x] = grad[y0:y0+win_h, x]

    path = dynamic_programming_path(grad_win, w_dia=1.0, use_linear=True)

    y_precise = np.zeros(w, dtype=np.float64)
    for x in range(w):
        y0 = int(round(coarse_edge[x]))
        y0 = np.clip(y0, 0, h - win_h)
        y_precise[x] = y0 + path[x]

    # 平滑
    y_precise = gaussian_filter1d(y_precise, sigma=w/40)
    return y_precise

# ================= LIMBUS / CSJ 检测 =================

def estimate_limbus_candidates_M2(image, exclude_center_px=200, smooth_sigma=15):
    """
    用 second central moment 方法（WTW 思路）估计左右 limbus 大致位置。
    仅用于给拟合提供初始区间。
    """
    h, w = image.shape
    M2 = np.zeros(w)

    ys = np.arange(h, dtype=np.float64)

    for x in range(w):
        col = image[:, x].astype(np.float64)
        col = col - col.min()
        s = col.sum()
        if s <= 0:
            M2[x] = 0
        else:
            p = col / s
            mu = (ys * p).sum()
            var = ((ys - mu)**2 * p).sum()
            M2[x] = var

    M2_s = gaussian_filter1d(M2, sigma=smooth_sigma)
    dM2 = np.gradient(M2_s)

    center = w // 2

    # 左右分别在远离中心的区域找导数极值
    left_region = dM2[:max(center - exclude_center_px, 10)]
    right_region = dM2[min(center + exclude_center_px, w-10):]

    if left_region.size == 0 or right_region.size == 0:
        # 回退：简单用 20% 和 80% 位置
        return int(w * 0.2), int(w * 0.8)

    # 取绝对值最大点（变化最剧烈）
    left_idx = int(np.argmax(np.abs(left_region)))
    right_idx = int(np.argmax(np.abs(right_region))) + min(center + exclude_center_px, w-10)

    return left_idx, right_idx

def fit_poly(x, y, deg):
    p = Polynomial.fit(x, y, deg)
    return p.convert()  # 转成标准多项式，方便求值

def refine_limbus_points_from_fit(x, y_surf, xL0, xR0, fit_win_px=150):
    """
    用 4阶多项式拟合中央角膜，用2阶拟合外侧巩膜，
    在左右各自的过渡区求交点作为 limbus。
    """
    w = len(x)
    center = w // 2

    # 中央角膜区：以中心为主
    cornea_mask = (x > center - fit_win_px) & (x < center + fit_win_px)
    p_cornea = fit_poly(x[cornea_mask], y_surf[cornea_mask], deg=4)

    # 左巩膜区：以左候选点为中心向外
    left_mask = (x >= max(0, xL0 - fit_win_px)) & (x <= xL0 + 20)
    p_sclera_L = fit_poly(x[left_mask], y_surf[left_mask], deg=2)

    # 右巩膜区
    right_mask = (x <= min(w-1, xR0 + fit_win_px)) & (x >= xR0 - 20)
    p_sclera_R = fit_poly(x[right_mask], y_surf[right_mask], deg=2)

    # 解交点：p_cornea(x) = p_sclera(x)
    def intersect(p1, p2):
        # p1 - p2 = 0
        pr = p1 - p2
        roots = pr.roots()
        # 只保留实根
        roots = np.real(roots[np.isreal(roots)])
        return roots

    # 左 limbus: 选在 xL0 附近的根
    rootsL = intersect(p_cornea, p_sclera_L)
    if len(rootsL) == 0:
        xL = xL0
    else:
        xL = rootsL[np.argmin(np.abs(rootsL - xL0))]
    xL = float(np.clip(xL, 0, w-1))
    yL = float(p_cornea(xL))

    # 右 limbus
    rootsR = intersect(p_cornea, p_sclera_R)
    if len(rootsR) == 0:
        xR = xR0
    else:
        xR = rootsR[np.argmin(np.abs(rootsR - xR0))]
    xR = float(np.clip(xR, 0, w-1))
    yR = float(p_cornea(xR))

    return (xL, yL), (xR, yR), p_cornea, p_sclera_L, p_sclera_R

# ================= 矢高计算 =================

def compute_mm_profile(apex_x, apex_y, y_surf):
    """
    把像素坐标转成以 apex 为原点的 (x_mm, z_mm).
    """
    w = len(y_surf)
    xs = np.arange(w)
    x_mm = (xs - apex_x) * REALW
    z_mm = (apex_y - y_surf) * REALH
    return x_mm, z_mm

def sagittal_depth(x_mm, z_mm, chord_len_mm):
    """
    以 apex 为弦中点、给定弦长 C，计算矢高。
    """
    C = chord_len_mm
    if C <= 0:
        return np.nan

    xL = -C / 2.0
    xR =  C / 2.0

    # 如果 chord 超出数据范围，返回 NaN
    if xL < x_mm.min() or xR > x_mm.max():
        return np.nan

    # 插值 z(x)
    z_interp = np.interp([xL, 0.0, xR], x_mm, z_mm)
    zL, z0, zR = z_interp

    # 弦线 z_line(x) 通过 (xL, zL), (xR, zR)
    # z_line(x) = zL + (zR - zL)/(xR - xL) * (x - xL)
    slope = (zR - zL) / (xR - xL)
    z_line_0 = zL + slope * (0.0 - xL)

    # 矢高 = 弦线在 x=0 的高度 - 曲线在 x=0 的高度(z0)
    return float(z_line_0 - z0)

def sagittal_depth_between_points(x_mm, z_mm, x1_mm, x2_mm):
    """
    通用版本：给定两个弦端点的 x 坐标（以 apex 为0），从真实前表面取 z，
    计算此弦相对 apex 的矢高。
    """
    if x1_mm == x2_mm:
        return np.nan
    xL, xR = sorted([x1_mm, x2_mm])

    if xL < x_mm.min() or xR > x_mm.max():
        return np.nan

    zL, z0, zR = np.interp([xL, 0.0, xR], x_mm, z_mm)

    slope = (zR - zL) / (xR - xL)
    z_line_0 = zL + slope * (0.0 - xL)
    return float(z_line_0 - z0)

# ================= 主流程 =================

def process_oct_image(path, debug_plot=True):
    # # 1. 读图
    # img = io.imread(path)
    # if img.ndim == 3:
    #     img = img[..., 0]
    # img = img.astype(np.float64)
    #
    # # 2. 预处理
    # img_corr = remove_HA(img)
    # # 可适当增强对比
    # img_corr = exposure.equalize_adapthist(img_corr, clip_limit=0.01)
    img = io.imread(path)
    if img.ndim == 3:
        img = img[..., 0]
    img = img.astype(np.float64)

    # 2. 预处理
    img_corr = remove_HA(img)

    # 把值映射到 [0,1]，避免负值和越界
    img_corr = img_corr - img_corr.min()
    max_val = img_corr.max()
    if max_val > 0:
        img_corr = img_corr / max_val
    else:
        img_corr[:] = 0.0

    # 自适应直方图均衡，输入现在是 [0,1] 的 float，合法
    img_corr = exposure.equalize_adapthist(img_corr, clip_limit=0.01)

    # 3. 梯度 & ICL_mask
    grad = gradient_image_vertical(img_corr, scale=0.4, option=1)
    mask = create_mask_from_gradient(grad)

    # 4. 粗前表面
    y_coarse = get_coarse_front_edge(mask)

    # 5. 精确前表面（DP）
    y_surf = refine_front_surface(grad, mask, y_coarse, win_frac=0.1)

    h, w = img.shape
    xs = np.arange(w)

    # 6. 顶点 apex（在中间1/3范围内取最浅点）
    center_region = (xs > w/3) & (xs < 2*w/3)
    idx_apex = center_region.nonzero()[0][np.argmin(y_surf[center_region])]
    x_apex = float(idx_apex)
    y_apex = float(y_surf[idx_apex])

    # 7. 初始 limbus 候选 (M2 方法)
    xL0, xR0 = estimate_limbus_candidates_M2(img_corr, exclude_center_px=int(w*0.2),
                                             smooth_sigma=15)

    # 8. 拟合 + 精化 limbus
    (xL, yL), (xR, yR), p_cornea, p_sL, p_sR = refine_limbus_points_from_fit(xs, y_surf, xL0, xR0,
                                                                             fit_win_px=int(w*0.15))

    # 9. mm 坐标
    x_mm, z_mm = compute_mm_profile(x_apex, y_apex, y_surf)

    # 10. CSJ (以 limbus-limbus chord 为弦)
    xL_mm = (xL - x_apex) * REALW
    xR_mm = (xR - x_apex) * REALW
    CSJ = sagittal_depth_between_points(x_mm, z_mm, xL_mm, xR_mm)

    # 11. CS10 （10 mm chord，中心在 apex）
    CS10 = sagittal_depth(x_mm, z_mm, chord_len_mm=10.0)

    # 12. 也给一个“全角膜 CS”：用 limbus chord（与 CSJ 同义，这里一起给出）
    CS_full = CSJ

    results = {
        "apex_px": (x_apex, y_apex),
        "limbus_left_px": (xL, yL),
        "limbus_right_px": (xR, yR),
        "apex_mm": (0.0, 0.0),
        "limbus_left_mm": (xL_mm, (y_apex - yL) * REALH),
        "limbus_right_mm": (xR_mm, (y_apex - yR) * REALH),
        "CSJ_mm": CSJ,
        "CS10_mm": CS10,
        "CS_full_mm": CS_full,
        "REALW_mm_per_px": REALW,
        "REALH_mm_per_px": REALH
    }

    if debug_plot:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.imshow(img, cmap='gray')
        ax.plot(xs, y_surf, 'r-', linewidth=1.0, label='Anterior surface')
        ax.scatter([x_apex], [y_apex], c='y', s=30, label='Apex')

        ax.scatter([xL, xR], [yL, yR], c='cyan', s=30, label='Limbus (CSJ)')

        # 画 CS10 chord
        if not np.isnan(CS10):
            C10 = 10.0
            xL10_mm, xR10_mm = -C10/2, C10/2
            xL10_px = x_apex + xL10_mm / REALW
            xR10_px = x_apex + xR10_mm / REALW
            zL10, z0_10, zR10 = np.interp([xL10_mm, 0.0, xR10_mm], x_mm, z_mm)
            # 转回像素坐标画弦线
            # z = (y_apex - y) * REALH => y = y_apex - z/REALH
            slope = (zR10 - zL10) / (xR10_mm - xL10_mm)
            # 在 [-C10/2, C10/2] 上画
            xs10_mm = np.linspace(xL10_mm, xR10_mm, 200)
            z_line10 = zL10 + slope * (xs10_mm - xL10_mm)
            ys_line10 = y_apex - z_line10 / REALH
            xs10_px = x_apex + xs10_mm / REALW
            ax.plot(xs10_px, ys_line10, 'g--', linewidth=0.8, label='Chord 10mm')

        ax.set_title("OCT anterior surface & automatic limbus / CS")
        ax.set_xlim(0, w-1)
        ax.set_ylim(h-1, 0)
        ax.legend(loc='lower right', fontsize=7)
        plt.tight_layout()
        plt.show()

    return results

if __name__ == "__main__":
    # 示例：直接处理你给的那张图
    path = r"D:\deeplearning\WTW\TSJ_OD_B_img005.tif"  # 换成你的图像路径
    res = process_oct_image(path, debug_plot=True)
    for k, v in res.items():
        print(k, ":", v)
