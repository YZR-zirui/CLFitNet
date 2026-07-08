# -*- coding: utf-8 -*-
# batch_cs_from_limbus_qc.py
import os, csv, numpy as np, matplotlib.pyplot as plt
from skimage import io, exposure, morphology, filters
from skimage.filters.rank import entropy
from skimage.morphology import disk
from scipy import ndimage
from scipy.ndimage import gaussian_filter1d

# ===== 配置 =====
IMAGE_FOLDER = r"D:\deeplearning\WTW\seg_ak"   # 原图或分割图（与 A 的 file 列对应）
CSV_IN       = r"D:\deeplearning\WTW\results\WTW\limbus_wtw_select_results.csv"
CSV_OUT      = r"D:\deeplearning\WTW\results\CS\cs_results.csv"
SAVE_PREV    = True
PREV_DIR     = r"D:\deeplearning\WTW\results\CS\preview"

SCAN_W_MM = 16.0
SCAN_H_MM = 11.989

# 强条件阈值
FOOT_T_MIN, FOOT_T_MAX = 0.05, 0.95     # 垂足在线段内
ALLOW_CS10 = True
# =================

# ---------- 工具 ----------
def remove_HA(img):
    img = img.astype(np.float64)
    return img - img.mean(axis=1, keepdims=True)

def gradient_vertical(I, scale=0.4, option=1):
    from skimage.transform import rescale, resize
    small = rescale(I, scale, anti_aliasing=True, preserve_range=True).astype(np.float64)
    se = np.array([[1.0],[-1.0]]) if option==1 else np.array([[-1.0],[1.0]])
    g  = ndimage.convolve(small, se, mode='nearest')
    g  = np.abs(g)
    g  = resize(g, I.shape, anti_aliasing=True, preserve_range=True)
    m  = g.max() if g.max()>0 else 1.0
    return g/m

def create_mask_from_gradient(grad):
    g8 = (grad*255).astype(np.uint8)
    ent = entropy(g8, disk(15)); ent = (ent/ent.max()*255).astype(np.uint8) if ent.max()>0 else ent
    thr = max(filters.threshold_otsu(ent.astype(np.float64)/255.0)-0.05, 0.0)
    bw  = (ent.astype(np.float64)/255.0) > thr
    bw  = morphology.closing(bw, morphology.disk(3))
    bw  = ndimage.binary_fill_holes(bw)
    bw  = morphology.remove_small_objects(bw, min_size=10000)
    bw  = morphology.opening(bw, morphology.disk(6))
    bw  = morphology.erosion(bw, morphology.disk(3))
    return bw.astype(bool)

def coarse_edge_from_mask(mask):
    h,w = mask.shape
    edges = morphology.binary_dilation(mask) ^ mask
    y = np.full(w, np.nan)
    for x in range(w):
        idx = np.where(edges[:,x])[0]
        if idx.size: y[x]=idx.min()
    valid = ~np.isnan(y)
    if valid.sum()<10:
        y = np.full(w, np.nan)
        for x in range(w):
            idx = np.where(mask[:,x])[0]
            if idx.size: y[x]=idx.min()
        valid = ~np.isnan(y)
        if valid.sum()<10: raise RuntimeError("Coarse edge failed.")
    return np.interp(np.arange(w), np.where(valid)[0], y[valid])

def dp_path(grad_win, w_dia=1.0):
    def linear_trans(x):
        xa,xb=0.6,0.8; ya,yb=0.1,0.8
        y=np.zeros_like(x)
        m1=x<=xa; y[m1]=ya/xa*x[m1]
        m2=(x>xa)&(x<=xb); y[m2]=(yb-ya)/(xb-xa)*x[m2]+(ya*xb-xa*yb)/(xb-xa)
        m3=x>xb; y[m3]=(yb-1)/(xb-1)*x[m3]+(xb-yb)/(xb-1)
        return y
    I = grad_win.astype(np.float64); I = I/(I.max() if I.max()>0 else 1.0)
    I = linear_trans(I)
    H,W = I.shape; l=np.zeros((H,W)); paths=np.zeros((H,W),dtype=np.int32)
    for j in range(1,W):
        s1=(2-I[0,j-1]-I[0,j])+l[0,j-1]
        s2=(2-I[1,j-1]-I[0,j])*w_dia+l[1,j-1]
        if s1<s2: l[0,j],paths[0,j]=s1,0
        else:     l[0,j],paths[0,j]=s2,1
        for i in range(1,H-1):
            c=[(2-I[i-1,j-1]-I[i,j])*w_dia+l[i-1,j-1],
               (2-I[i  ,j-1]-I[i,j])        +l[i  ,j-1],
               (2-I[i+1,j-1]-I[i,j])*w_dia+l[i+1,j-1]]
            p=int(np.argmin(c)); l[i,j]=c[p]; paths[i,j]=i+p-1
        s1=(2-I[H-2,j-1]-I[H-1,j])*w_dia+l[H-2,j-1]
        s2=(2-I[H-1,j-1]-I[H-1,j])+l[H-1,j-1]
        if s1<s2: l[H-1,j],paths[H-1,j]=s1,H-2
        else:     l[H-1,j],paths[H-1,j]=s2,H-1
    path=np.zeros(W,dtype=np.int32); path[-1]=int(np.argmin(l[:,-1]))
    for j in range(W-1,0,-1): path[j-1]=paths[path[j],j]
    return path

def refine_surface(grad, coarse, win_frac=0.1):
    h,w = grad.shape; win_h=max(int(h*win_frac),20)
    gw=np.zeros((win_h,w),np.float64)
    for x in range(w):
        y0=int(round(coarse[x])); y0=np.clip(y0,0,h-win_h); gw[:,x]=grad[y0:y0+win_h,x]
    path=dp_path(gw); y=np.zeros(w)
    for x in range(w):
        y0=int(round(coarse[x])); y0=np.clip(y0,0,h-win_h); y[x]=y0+path[x]
    return gaussian_filter1d(y, sigma=w/40)

def extract_surface_from_seg(seg):
    H,W=seg.shape; vals,cnts=np.unique(seg,return_counts=True); bg=int(vals[np.argmax(cnts)])
    y=np.full(W,np.nan)
    for x in range(W):
        col=seg[:,x]; idx=np.where(col!=bg)[0]
        if idx.size: y[x]=idx[0]
    valid=~np.isnan(y)
    if valid.sum()<3: return None
    return np.interp(np.arange(W), np.where(valid)[0], y[valid])

def anterior_surface(image_path):
    img=io.imread(image_path)
    if img.ndim==3: img=img[...,0]
    img=img.astype(np.float64)
    # 分割图：列顶点
    if len(np.unique(img))<32:
        y=extract_surface_from_seg(img.astype(np.uint8))
        if y is not None: return y, img.shape
    # 原图：梯度 + DP
    imgc=remove_HA(img); imgc-=imgc.min(); m=imgc.max(); imgc/= (m if m>0 else 1.0)
    imgc=exposure.equalize_adapthist(imgc,clip_limit=0.01)
    grad=gradient_vertical(imgc); mask=create_mask_from_gradient(grad)
    coarse=coarse_edge_from_mask(mask); y=refine_surface(grad,coarse)
    return y, img.shape

# === 几何 ===
def apex_to_chord_distance_and_foot(xL,zL,xR,zR):
    A=zR-zL; B=xL-xR; C=xR*zL - xL*zR
    denom = A*A + B*B
    if denom==0: return np.nan, np.nan, np.nan, np.nan
    # 垂距
    dist = abs(C)/np.sqrt(denom)
    # 垂足坐标（apex=(0,0)）
    xf = -A*C/denom; zf = -B*C/denom
    # 垂足参数 t（0=L,1=R），用于“是否在线段内”的 QC
    # 用投影参数： t = ((P-L)·(R-L))/||R-L||^2；此处 P=垂足
    dx = xR-xL; dz = zR-zL; denom2 = dx*dx + dz*dz
    t = ((xf-xL)*dx + (zf-zL)*dz)/denom2 if denom2>0 else np.nan
    return float(dist), float(xf), float(zf), float(t)

def sagittal_depth_from_profile(x_mm, z_mm, chord_len_mm=10.0):
    C=chord_len_mm; xL=-C/2; xR=C/2
    if xL<x_mm.min() or xR>x_mm.max(): return np.nan  # 视野不够
    zL, z0, zR = np.interp([xL, 0.0, xR], x_mm, z_mm)
    slope = (zR - zL) / (xR - xL)
    z_line_0 = zL + slope * (0.0 - xL)
    return float(abs(z_line_0 - z0))  # 报告用“非负长度”

def f2n(v):
    try:
        if v is None: return None
        s=str(v).strip().lower()
        if s in ("","na","nan","none","null"): return None
        x=float(s)
        if np.isnan(x) or np.isinf(x): return None
        return x
    except: return None

def process_batch(image_folder, csv_in, csv_out, save_prev=True, prev_dir=None, want_cs10=True):
    if save_prev:
        prev_dir = prev_dir or os.path.join(image_folder,"cs_preview")
        os.makedirs(prev_dir, exist_ok=True)
    os.makedirs(os.path.dirname(csv_out) if os.path.dirname(csv_out) else ".", exist_ok=True)

    rows=list(csv.DictReader(open(csv_in,'r',encoding='utf-8-sig')))
    results=[]
    for i,r in enumerate(rows,1):
        fname = r.get("file") or r.get("filename") or r.get("name")
        if not fname:
            print(f"[{i}] 跳过：CSV 缺 file 列"); continue
        img_path = os.path.join(image_folder, fname)
        if not os.path.exists(img_path):
            print(f"[{i}] 跳过：找不到图像 {img_path}"); continue

        # limbus 像素坐标（或毫米兜底）
        xL_px,yL_px,xR_px,yR_px = f2n(r.get("xL_px")),f2n(r.get("yL_px")),f2n(r.get("xR_px")),f2n(r.get("yR_px"))
        xL_mm_csv,zL_mm_csv,xR_mm_csv,zR_mm_csv = f2n(r.get("xL_mm")),f2n(r.get("zL_mm")),f2n(r.get("xR_mm")),f2n(r.get("zR_mm"))

        # 前表面 & apex
        try:
            y_surf,(H,W) = anterior_surface(img_path)
        except Exception as e:
            print(f"[{i}] NA：前表面失败 {fname}: {e}")
            continue

        realw = SCAN_W_MM/(W-1); realh = SCAN_H_MM/H
        xs = np.arange(W)
        mid = (xs>W/3)&(xs<2*W/3)
        idx = mid.nonzero()[0][np.argmin(y_surf[mid])]
        xA, yA = float(idx), float(y_surf[idx])

        # 像素缺失则毫米→像素回推
        if None in (xL_px,yL_px,xR_px,yR_px):
            if None not in (xL_mm_csv,zL_mm_csv,xR_mm_csv,zR_mm_csv):
                xL_px = xA + xL_mm_csv/realw; yL_px = yA - zL_mm_csv/realh
                xR_px = xA + xR_mm_csv/realw; yR_px = yA - zR_mm_csv/realh
            else:
                print(f"[{i}] NA：缺 Limbus 坐标 {fname}")
                continue

        # APEX 原点毫米坐标
        xL_mm = (xL_px - xA)*realw; zL_mm = (yA - yL_px)*realh
        xR_mm = (xR_px - xA)*realw; zR_mm = (yA - yR_px)*realh
        x_mm   = (xs - xA)*realw;   z_mm   = (yA - y_surf)*realh

        # CSJ + 垂足参数 t
        CSJ_mm, foot_x_mm, foot_z_mm, t_param = apex_to_chord_distance_and_foot(xL_mm,zL_mm,xR_mm,zR_mm)

        qc_reason=[]
        if not (FOOT_T_MIN <= t_param <= FOOT_T_MAX):
            qc_reason.append("foot_outside_chord")

        # CS10（可选）
        if want_cs10:
            CS10_mm = sagittal_depth_from_profile(x_mm, z_mm, chord_len_mm=10.0)
            if np.isnan(CS10_mm): qc_reason.append("FOV_insufficient_for_CS10")
        else:
            CS10_mm = None

        # WTW（以 APEX 坐标计算，等价于 A 步）
        WTW_mm = float(np.hypot(xR_mm-xL_mm, zR_mm-zL_mm))

        rec = {"file":fname,
               "apex_x_px":xA, "apex_y_px":yA,
               "xL_px":xL_px, "yL_px":yL_px, "xR_px":xR_px, "yR_px":yR_px,
               "xL_mm":xL_mm, "zL_mm":zL_mm, "xR_mm":xR_mm, "zR_mm":zR_mm,
               "WTW_mm":WTW_mm, "CSJ_mm":CSJ_mm, "CS10_mm":CS10_mm,
               "foot_t":t_param, "qc_reason":"|".join(qc_reason)}
        results.append(rec)

        # 预览
        if save_prev:
            try:
                img=io.imread(img_path)
                if img.ndim==3: img=img[...,0]
                plt.figure(figsize=(6,4))
                plt.imshow(img, cmap='gray', origin='upper')
                plt.plot(xs, y_surf, 'r-', lw=1.0, label='Anterior surface')
                plt.scatter([xA],[yA], c='y', s=30, label='Apex')
                plt.scatter([xL_px,xR_px],[yL_px,yR_px], c='c', s=30, label='Limbus')
                # 画弦
                slope = (zR_mm-zL_mm)/(xR_mm-xL_mm) if not np.isclose(xL_mm,xR_mm) else 0.0
                xm = np.linspace(min(xL_mm,xR_mm)-2, max(xL_mm,xR_mm)+2, 200)
                zm = zL_mm + slope*(xm - xL_mm)
                xp = xA + xm/realw; yp = yA - zm/realh
                plt.plot(xp, yp, 'g--', lw=0.8, label='Chord')
                # 垂线
                fx = xA + foot_x_mm/realw; fz = yA - foot_z_mm/realh
                plt.plot([xA,fx],[yA,fz], 'w-', lw=1.3, label=f'CSJ={CSJ_mm:.3f} mm')
                ttl = f"{fname}  WTW={WTW_mm:.3f}  CSJ={CSJ_mm:.3f}"
                ttl+= f"  CS10={CS10_mm:.3f}" if (CS10_mm is not None and not np.isnan(CS10_mm)) else "  CS10=NA"
                if qc_reason: ttl += f"  [{','.join(qc_reason)}]"
                plt.title(ttl); plt.legend(fontsize=7, loc='lower right')
                Hh,Ww=img.shape[:2]; plt.xlim(0,Ww-1); plt.ylim(Hh-1,0); plt.tight_layout()
                os.makedirs(prev_dir, exist_ok=True)
                plt.savefig(os.path.join(prev_dir, os.path.splitext(fname)[0]+"_cs.png"), dpi=160)
                plt.close()
            except Exception as e:
                print(f"[{i}] 预览失败 {fname}: {e}")

        msg = f"[{i}/{len(rows)}] {fname}  WTW={WTW_mm:.3f}  CSJ={CSJ_mm:.3f}  "
        msg+= (f"CS10={CS10_mm:.3f}" if (CS10_mm is not None and not np.isnan(CS10_mm)) else "CS10=NA")
        if qc_reason: msg+= f"  -> {qc_reason}"
        print(msg)

    # 输出 CSV
    fields=["file","apex_x_px","apex_y_px","xL_px","yL_px","xR_px","yR_px",
            "xL_mm","zL_mm","xR_mm","zR_mm","WTW_mm","CSJ_mm","CS10_mm","foot_t","qc_reason"]
    os.makedirs(os.path.dirname(CSV_OUT) if os.path.dirname(CSV_OUT) else ".", exist_ok=True)
    with open(CSV_OUT,'w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f, fieldnames=fields); w.writeheader()
        for r in results:
            row={}
            for k,v in r.items():
                if isinstance(v,(float,np.floating)):
                    row[k]=None if (np.isnan(v) or np.isinf(v)) else f"{v:.6f}"
                else:
                    row[k]=v
            w.writerow(row)
    print("\nCS 结果已写入：", CSV_OUT)
    if save_prev: print("预览目录：", PREV_DIR)

if __name__=="__main__":
    process_batch(IMAGE_FOLDER, CSV_IN, CSV_OUT, SAVE_PREV, PREV_DIR, want_cs10=ALLOW_CS10)
