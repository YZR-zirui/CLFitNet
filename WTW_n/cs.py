import numpy as np
import matplotlib.pyplot as plt
from skimage import io, exposure, morphology, filters
from skimage.filters.rank import entropy
from skimage.morphology import disk
from scipy import ndimage
from scipy.ndimage import gaussian_filter1d

# ================= 标定（按你提供） =================
REALW = 16.0 / 1160.0                  # mm/px (horizontal)
REALH = 11.0 / 796.0 / 1.376           # mm/px (vertical, refractive correction)

# ================ 基础函数：前表面提取（精简版） =================
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

# =============== 坐标/矢高计算 =================
def pixels_to_mm_x(x_px, apex_x_px):
    return (x_px - apex_x_px) * REALW

def pixels_to_mm_z(y_px, apex_y_px):
    # z = (apex_y - y) * REALH, 上方为正
    return (apex_y_px - y_px) * REALH

def vertical_sag_mm_from_line_and_apex(line_pts_mm, apex_mm):
    """line_pts_mm: [(x1_mm, z1_mm), (x2_mm, z2_mm)], apex_mm: (0,0) 已经在 apex-相对坐标系"""
    (x1, z1), (x2, z2) = line_pts_mm
    # 直线通过两点：z = z1 + (z2-z1)/(x2-x1)*(x-x1)
    if np.isclose(x1, x2):
        # 极端情况：弦线竖直，取 apex 的 x=0 处 z_line = z1 (或 z2)，vertical sag = z_line(0) - z(0)=z_line(0)
        return float(z1)
    slope = (z2 - z1) / (x2 - x1)
    z_line_at_0 = z1 + slope * (0.0 - x1)
    # apex 在原点 z(0)=0
    return float(z_line_at_0)

def perpendicular_sag_mm_from_line_and_apex(line_pts_mm, apex_mm):
    """apex 取 (0,0)；返回 apex 到直线的垂距"""
    (x1, z1), (x2, z2) = line_pts_mm
    A = z2 - z1
    B = x1 - x2
    C = x2*z1 - x1*z2
    # 垂距 = |A*0 + B*0 + C| / sqrt(A^2 + B^2)
    denom = np.hypot(A, B)
    if denom == 0:
        return np.nan
    return float(abs(C) / denom)

# =============== 主函数：你只需给 Limbus 点 ==============
def compute_sag_with_given_limbus(image_path, limbus_left_px, limbus_right_px,
                                  plot=True):
    """
    limbus_left_px/right_px: (x_px, y_px)
    返回：
      - apex 像素坐标
      - CSJ 垂直矢高（mm）
      - CSJ 垂直+垂直线可视化
      - CSJ 垂距（mm, 推荐）
    """
    # 读图
    img = io.imread(image_path)
    if img.ndim == 3:
        img = img[..., 0]
    img = img.astype(np.float64)

    # 预处理（把范围压到 [0,1] 再做自适应均衡）
    img_corr = remove_HA(img)
    img_corr = img_corr - img_corr.min()
    m = img_corr.max()
    img_corr = img_corr / (m if m > 0 else 1.0)
    img_corr = exposure.equalize_adapthist(img_corr, clip_limit=0.01)

    # 梯度 & mask & 前表面
    grad = gradient_vertical(img_corr, scale=0.4, option=1)
    mask = create_mask_from_gradient(grad)
    y_coarse = coarse_edge_from_mask(mask)
    y_surf = refine_surface(grad, y_coarse, win_frac=0.1)

    h, w = img.shape
    xs = np.arange(w)

    # apex：在中间 1/3 里 y 最小（最向上）
    center = (xs > w/3) & (xs < 2*w/3)
    idx_apex = center.nonzero()[0][np.argmin(y_surf[center])]
    x_apex, y_apex = float(idx_apex), float(y_surf[idx_apex])

    # 把点转成以 apex 为原点的 mm 坐标
    (xL_px, yL_px) = limbus_left_px
    (xR_px, yR_px) = limbus_right_px

    xL_mm = pixels_to_mm_x(xL_px, x_apex)
    zL_mm = pixels_to_mm_z(yL_px, y_apex)
    xR_mm = pixels_to_mm_x(xR_px, x_apex)
    zR_mm = pixels_to_mm_z(yR_px, y_apex)

    # 垂直/垂距矢高
    line_pts_mm = [(xL_mm, zL_mm), (xR_mm, zR_mm)]
    apex_mm = (0.0, 0.0)

    CSJ_vertical_mm = vertical_sag_mm_from_line_and_apex(line_pts_mm, apex_mm)
    CSJ_perp_mm     = perpendicular_sag_mm_from_line_and_apex(line_pts_mm, apex_mm)

    out = {
        "apex_px": (x_apex, y_apex),
        "limbus_left_px": limbus_left_px,
        "limbus_right_px": limbus_right_px,
        "CSJ_vertical_mm": CSJ_vertical_mm,
        "CSJ_perpendicular_mm": CSJ_perp_mm,
        "REALW_mm_per_px": REALW,
        "REALH_mm_per_px": REALH,
    }

    if plot:
        # 为了画“垂直矢高”，需要画弦线并在 x=apex 处取交点
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.imshow(img, cmap='gray')
        ax.plot(xs, y_surf, 'r-', lw=1.0, label='Anterior surface')
        ax.scatter([x_apex], [y_apex], c='y', s=30, label='Apex')
        ax.scatter([xL_px, xR_px], [yL_px, yR_px], c='c', s=30, label='Limbus (input)')

        # 画通过两端点的弦线（在像素坐标系里画直线）
        # 把 mm 线转换回像素坐标更准确（避免像素各向异性带来的几何误差）
        # 线：z = zL + slope*(x_mm - xL_mm)
        # 转回像素：y = y_apex - z/REALH； x_px = x_apex + x_mm/REALW
        if not np.isnan(CSJ_vertical_mm):
            slope = (zR_mm - zL_mm) / (xR_mm - xL_mm) if not np.isclose(xR_mm, xL_mm) else 0.0
            x_mm_line = np.linspace(min(xL_mm, xR_mm)-2, max(xL_mm, xR_mm)+2, 200)
            z_mm_line = zL_mm + slope * (x_mm_line - xL_mm)
            x_px_line = x_apex + x_mm_line / REALW
            y_px_line = y_apex - z_mm_line / REALH
            ax.plot(x_px_line, y_px_line, 'g--', lw=0.8, label='Chord (through Limbus)')

            # 垂直矢高：在 x=apex 处的差
            z_line_at_0 = vertical_sag_mm_from_line_and_apex(line_pts_mm, apex_mm)
            y_line_at_apex = y_apex - z_line_at_0 / REALH
            ax.plot([x_apex, x_apex], [y_apex, y_line_at_apex], 'w-', lw=1.2,
                    label=f'Vertical sag = {CSJ_vertical_mm:.3f} mm')

        ax.set_title('CSJ from given Limbus points')
        ax.set_xlim(0, w-1)
        ax.set_ylim(h-1, 0)
        ax.legend(loc='lower right', fontsize=7)
        plt.tight_layout()
        plt.show()

    return out

# ------------- 示例 -------------
if __name__ == "__main__":
    # 例子：换成你的图像路径和你测得的 Limbus（像素）坐标
    image_path = "TSJ_OD_B_img005.tif"
    limbus_left_px  = (200, 250)   # TODO: 改成你的 (xL, yL)
    limbus_right_px = (940, 255)   # TODO: 改成你的 (xR, yR)

    res = compute_sag_with_given_limbus(image_path, limbus_left_px, limbus_right_px, plot=True)
    for k, v in res.items():
        print(f"{k}: {v}")
