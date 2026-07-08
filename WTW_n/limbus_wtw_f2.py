# -*- coding: utf-8 -*-
import os
import csv
import glob
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, morphology, measure
from scipy import ndimage

# ===================== 配置（请修改） =====================
IMG_DIR       = r"D:\deeplearning\WTW\seg_ak"      # 必填：图像/分割图所在文件夹
OUT_CSV       = r"D:\deeplearning\WTW\results\WTW\limbus_wtw_results.csv"      # 必填：输出 CSV 路径
SAVE_PREVIEW  = True     # 是否保存预览图（带 L/R limbus 与 WTW）
PREVIEW_DIR   = r"D:\deeplearning\WTW\results\WTW\preview"      # 留空则默认 IMG_DIR/wtw_preview

# 你的扫描物理尺寸（和之前一致）
SCAN_WIDTH_MM = 16.0
SCAN_DEPTH_MM = 11.989

# 读取哪些文件？（通配符）
GLOB_PATTERN  = "*.png"  # 例如 *.png / *_pred3.png / *.tif 等

# Limbus 类的标签号（若是二值掩膜则为1；若是多类分割，请改成 limbus 所在的类别号）
LIMBUS_LABEL  = 50
# =========================================================


# 统一的“记录并打印原因”小工具
def fail_reason(file, reason, results):
    results.append({
        "file": file,
        "xL_px": None, "yL_px": None,
        "xR_px": None, "yR_px": None,
        "xL_mm": None, "zL_mm": None,
        "xR_mm": None, "zR_mm": None,
        "WTW_mm": None,
        "Reason": reason
    })
    print(f"[NA] {file}  -> {reason}")


def load_seg_or_mask(path):
    """读入分割图/掩膜；返回 uint8 图（背景=0，前景=非0）与原始数组。
       若是多类分割，自动把 ==LIMBUS_LABEL 的像素置为1，其它为0。"""
    try:
        arr = io.imread(path)
    except Exception as e:
        return None, None, f"读取图像失败: {e}"

    if arr.ndim == 3:
        arr = arr[..., 0]  # 取单通道

    arr = np.asarray(arr)

    # 若像素值离散很少，可能是多类分割
    vals = np.unique(arr)
    if len(vals) <= 32:
        # 二值或多类：抽出 limbus 类
        mask = (arr == LIMBUS_LABEL).astype(np.uint8)
    else:
        # 连续灰度：假设已经是二值 limbus（0/255 或 0/1），否则请自行预处理
        mask = (arr > 0).astype(np.uint8)

    return mask, arr, None


def pick_two_limbus_points(mask):
    """
    从二值 limbus 掩膜里找左右两个点（像素坐标，返回 (xL,yL),(xR,yR)）。
    规则尽量“保守+稳定”：
      1) 形态学开/闭去噪；取最大的两个连通域（或最大的一个连通域的最左/最右极点）。
      2) 每个连通域的“上缘”更接近角膜前表面，这里取连通域里 y 最小的像素作为候选，
         左域取 x 最小者，右域取 x 最大者（再各自回到对应域里的 y 最小处）。
    """
    if mask is None or mask.sum() == 0:
        return None, None, "Limbus 掩膜为空"

    # 形态学清理
    mk = morphology.binary_opening(mask, morphology.disk(2))
    mk = morphology.binary_closing(mk, morphology.disk(3))
    mk = morphology.remove_small_objects(mk, min_size=500)
    mk = mk.astype(np.uint8)

    if mk.sum() == 0:
        return None, None, "Limbus 掩膜清理后为空（可能阈值或标签不对）"

    # 连通域
    lab = measure.label(mk, connectivity=2)
    props = measure.regionprops(lab)
    if len(props) == 0:
        return None, None, "未检测到连通域"

    # 按面积降序
    props = sorted(props, key=lambda p: p.area, reverse=True)

    # 如果连通域≥2，取面积最大的两块；否则用一块的最左/最右极点
    if len(props) >= 2:
        pL, pR = props[0], props[1]
        # 让 pL 真正在“左侧”
        if pL.centroid[1] > pR.centroid[1]:
            pL, pR = pR, pL
        coordsL = pL.coords
        coordsR = pR.coords
        # 各自 y 最小（越上方）
        yL = int(coordsL[:, 0].min())
        # 在该 y 行内取 x 最小更靠外
        xL = int(coordsL[coordsL[:, 0] == yL][:, 1].min())

        yR = int(coordsR[:, 0].min())
        xR = int(coordsR[coordsR[:, 0] == yR][:, 1].max())
    else:
        # 单一大连通域：直接取整块的最左极点与最右极点；并各自回到各自上缘
        coords = props[0].coords
        x_min = int(coords[:, 1].min())
        x_max = int(coords[:, 1].max())

        # 左极点附近的“上缘”
        ys_left = coords[coords[:, 1] == x_min][:, 0]
        if ys_left.size == 0:
            return None, None, "单连通域但无法找到左极点"
        yL = int(ys_left.min()); xL = x_min

        # 右极点附近的“上缘”
        ys_right = coords[coords[:, 1] == x_max][:, 0]
        if ys_right.size == 0:
            return None, None, "单连通域但无法找到右极点"
        yR = int(ys_right.min()); xR = x_max

    return (xL, yL), (xR, yR), None


def process_folder(img_dir, out_csv, pattern="*.png",
                   save_preview=True, preview_dir=None):

    if not preview_dir and save_preview:
        preview_dir = os.path.join(img_dir, "wtw_preview")
    if save_preview:
        os.makedirs(preview_dir, exist_ok=True)

    files = sorted(glob.glob(os.path.join(img_dir, pattern)))
    if not files:
        print("没有找到任何文件，请检查 GLOB_PATTERN 与 IMG_DIR。")
        return

    results = []
    for i, path in enumerate(files, 1):
        fname = os.path.basename(path)

        # 1) 读取分割/掩膜
        mask, raw, err = load_seg_or_mask(path)
        if err:
            fail_reason(fname, f"读取/解析分割失败：{err}", results)
            continue

        H, W = mask.shape[:2]
        realw = SCAN_WIDTH_MM / (W - 1)
        realh = SCAN_DEPTH_MM / H  # 这里用于可视化比例一致即可，WTW 只用到 realw/realh 组合计算

        # 2) 找左右 limbus 点
        (xL, yL), (xR, yR), err2 = pick_two_limbus_points(mask)
        if err2:
            fail_reason(fname, f"找 Limbus 失败：{err2}", results)
            continue

        # 3) 基本有效性检查
        if not (0 <= xL < W and 0 <= xR < W and 0 <= yL < H and 0 <= yR < H):
            fail_reason(fname, "Limbus 点越界（像素坐标超出图像范围）", results)
            continue

        # 4) 计算毫米坐标（以图像中心行/列为原点没有必要；WTW 只需两点距离）
        #    注意：WTW 是两 Limbus 点连线长度（单位：mm），这条线不一定水平，所以要用欧氏距离。
        dx_mm = (xR - xL) * realw
        dz_mm = (yL - yR) * realh  # y 轴向下，符号不重要，距离取绝对值
        WTW = float(np.hypot(dx_mm, dz_mm))

        # 5) 可选：再做一个“异常值”门限，太小/太大都提示原因
        if not (8.0 <= WTW <= 14.5):   # 你可以根据自己数据分布调整
            reason = f"WTW 异常值（{WTW:.2f} mm），请核对分割是否可靠"
        else:
            reason = ""

        # 6) 保存结果
        rec = {
            "file": fname,
            "xL_px": float(xL), "yL_px": float(yL),
            "xR_px": float(xR), "yR_px": float(yR),
            "xL_mm": float((xL) * realw), "zL_mm": float((H-1 - yL) * realh),  # 仅作为参考坐标
            "xR_mm": float((xR) * realw), "zR_mm": float((H-1 - yR) * realh),
            "WTW_mm": WTW,
            "Reason": reason
        }
        results.append(rec)

        # 7) 预览图
        if save_preview:
            try:
                fig = plt.figure(figsize=(6,4))
                plt.imshow(mask, cmap="gray")
                plt.scatter([xL, xR], [yL, yR], s=40, c=["cyan","cyan"], label="Limbus")
                plt.plot([xL, xR], [yL, yR], "g--", lw=1.2, label=f"WTW={WTW:.2f} mm")
                plt.gca().invert_yaxis()  # 让视觉习惯与OCT一致（上方为小 y）
                plt.legend(loc="lower right", fontsize=8)
                plt.title(fname)
                plt.tight_layout()
                out_png = os.path.join(preview_dir, os.path.splitext(fname)[0] + "_wtw.png")
                plt.savefig(out_png, dpi=160)
                plt.close(fig)
            except Exception as e:
                print(f"[warn] 预览图失败 {fname}: {e}")

        # 8) 控制台打印
        okmsg = f"[{i}/{len(files)}] {fname}  WTW={WTW:.3f}"
        if reason:
            okmsg += f"  ({reason})"
        print(okmsg)

    # 9) 写 CSV
    fieldnames = ["file",
                  "xL_px","yL_px","xR_px","yR_px",
                  "xL_mm","zL_mm","xR_mm","zR_mm",
                  "WTW_mm","Reason"]
    os.makedirs(os.path.dirname(out_csv) if os.path.dirname(out_csv) else ".", exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in results:
            w.writerow(r)

    print(f"\n已写出：{out_csv}")
    if SAVE_PREVIEW:
        print(f"预览图目录：{PREVIEW_DIR if PREVIEW_DIR else os.path.join(IMG_DIR, 'wtw_preview')}")

if __name__ == "__main__":
    if not IMG_DIR or not OUT_CSV:
        print("请先设置顶部的 IMG_DIR / OUT_CSV / GLOB_PATTERN 等参数")
    else:
        process_folder(IMG_DIR, OUT_CSV, pattern=GLOB_PATTERN,
                       save_preview=SAVE_PREVIEW,
                       preview_dir=PREVIEW_DIR if PREVIEW_DIR else None)

