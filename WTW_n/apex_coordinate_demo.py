import numpy as np
import matplotlib.pyplot as plt

# --- 合成一幅与 OCT 尺寸接近的画布 ---
W, H = 1160, 796                     # 图像尺寸（像素）
SCAN_W_MM, SCAN_H_MM = 16.0, 11.989  # 物理视野（毫米）
realw = SCAN_W_MM/(W-1)
realh = SCAN_H_MM/H

# --- 例子：APEX 与 Limbus 的像素坐标（可替换为你的真实结果）---
xA_px, yA_px = 580, 150
xL_px, yL_px = 110, 220
xR_px, yR_px = 1010, 230

# --- 像素 → 以 APEX 为原点的毫米坐标 ---
def px_to_mm_apex(x_px, y_px):
    x_mm = (x_px - xA_px) * realw        # 右正
    z_mm = (yA_px - y_px) * realh        # 上正
    return x_mm, z_mm

xL_mm, zL_mm = px_to_mm_apex(xL_px, yL_px)
xR_mm, zR_mm = px_to_mm_apex(xR_px, yR_px)

# --- Limbus 弦 & CSJ ---
A = (zR_mm - zL_mm)
B = (xL_mm - xR_mm)
C = (xR_mm * zL_mm - xL_mm * zR_mm)
den = A*A + B*B
CSJ_mm = abs(C) / np.sqrt(den)
xf_mm = -A*C/den
zf_mm = -B*C/den

# 毫米 → 像素（画图用）
xf_px = xA_px + xf_mm/realw
zf_px = yA_px - zf_mm/realh

# --- 画图：用像素坐标显示（与原图方向一致）---
fig, ax = plt.subplots(figsize=(8, 5))
# 画一个矩形边框表示图像范围
ax.add_patch(plt.Rectangle((0,0), W, H, fill=False, linewidth=1))

# APEX & Limbus
ax.plot(xA_px, yA_px, 'o', markersize=7, label='APEX (pixel)')
ax.plot([xL_px, xR_px], [yL_px, yR_px], 'o', markersize=6, label='Limbus (pixel)')

# 画 Limbus 弦：在毫米直线上取点，再映射回像素
xm = np.linspace(min(xL_mm, xR_mm) - 2, max(xL_mm, xR_mm) + 2, 200)
slope = (zR_mm - zL_mm) / (xR_mm - xL_mm)
zm = zL_mm + slope*(xm - xL_mm)
xp = xA_px + xm/realw
yp = yA_px - zm/realh
ax.plot(xp, yp, '--', linewidth=1, label='Limbus chord')

# APEX 到弦的垂线（CSJ）
ax.plot([xA_px, xf_px], [yA_px, zf_px], '-', linewidth=1.5, label=f'CSJ = {CSJ_mm:.3f} mm')

# 像素坐标轴方向提示
ax.annotate('x (pixel) →', xy=(W*0.8, H*0.05), xytext=(W*0.6, H*0.05),
            arrowprops=dict(arrowstyle='->'), ha='center', va='center')
ax.annotate('y (pixel) ↓', xy=(W*0.05, H*0.3), xytext=(W*0.05, H*0.1),
            arrowprops=dict(arrowstyle='->'), ha='center', va='center')

# 公式说明
text = (f"Apex-centered mm coords:\n"
        f"x_mm = (x_px - xA_px) * {realw:.6f}\n"
        f"z_mm = (yA_px - y_px) * {realh:.6f}\n"
        f"L(mm)=({xL_mm:.2f},{zL_mm:.2f}), R(mm)=({xR_mm:.2f},{zR_mm:.2f})\n"
        f"CSJ = |C|/sqrt(A^2+B^2),  A=zR-zL, B=xL-xR, C=xR*zL - xL*zR")
ax.text(W*0.02, H*0.97, text, fontsize=9, va='top')

ax.set_xlim(0, W)
ax.set_ylim(H, 0)   # 上方为 0（与原图一致）
ax.set_title("Pixel coords (top-left origin) with APEX-centered mm overlay")
ax.legend(loc='lower right', fontsize=8)
plt.tight_layout()
plt.savefig("apex_coordinate_demo.png", dpi=160)
print("Saved: apex_coordinate_demo.png")
