import os
import glob
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from tqdm import tqdm

class ThicknessLoss(nn.Module):
    def __init__(self, k=5, scale=1.0):
        """
        可微动态厚度损失 (Differentiable Dynamic Thickness Loss)
        :param k: 膨胀核大小 (用于确定边缘感知的 ROI 区域)
        :param scale: 物理尺度转换 (um/pix)，默认为1.0即计算像素厚度损失
        """
        super(ThicknessLoss, self).__init__()
        self.k = k
        self.scale = scale
        # Step 1: 垂直方向的 Sobel 算子
        sobel_y = torch.tensor([[-1., -2., -1.], 
                                [ 0.,  0.,  0.], 
                                [ 1.,  2.,  1.]], dtype=torch.float32).view(1, 1, 3, 3)
        self.register_buffer('sobel_y', sobel_y)

    def forward(self, pred, mask):
        """
        :param pred: 预测的生成图像 [B, 1, H, W]
        :param mask: 对应的泪液层真实掩码 [B, 1, H, W]
        """
        B, C, H, W = pred.shape
        eps = 1e-6 # 防止除以零
        
        # 1. 计算预测图的垂直梯度
        grad_y = F.conv2d(pred, self.sobel_y, padding=1)
        
        # 2. 膨胀边缘掩码区 M_ROI = Dilate(M_3, k)
        M_ROI = F.max_pool2d(mask, kernel_size=self.k, stride=1, padding=self.k//2)
        
        # 3. 提取目标梯度
        G_focus = grad_y * M_ROI
        
        # 4. 分离上下边
        E_up = F.relu(G_focus)
        E_down = F.relu(-G_focus)
        
        # 5. 计算垂直质心 (Soft-argmax)
        y_coords = torch.arange(H, device=pred.device, dtype=torch.float32).view(1, 1, H, 1)
        
        sum_E_up = torch.sum(E_up, dim=2, keepdim=True) + eps
        sum_E_down = torch.sum(E_down, dim=2, keepdim=True) + eps
        
        y_bar_up = torch.sum(y_coords * E_up, dim=2, keepdim=True) / sum_E_up
        y_bar_down = torch.sum(y_coords * E_down, dim=2, keepdim=True) / sum_E_down
        
        # 6. 计算软厚度
        T_soft = y_bar_down - y_bar_up
        
        # 7. 计算真实厚度 T_gt
        T_gt = torch.sum(mask, dim=2, keepdim=True) 
        
        # 8. 转换成临床物理单位 (um)
        T_um_pred = T_soft * self.scale
        T_um_gt = T_gt * self.scale
        
        # 9. 计算绝对值误差
        valid_cols = (T_gt > 0).float()
        loss = torch.abs(T_um_pred - T_um_gt) * valid_cols
        
        L_thick = torch.sum(loss) / (torch.sum(valid_cols) + eps)
        
        return L_thick

def load_image_as_tensor(img_path, is_mask=False, device='cpu'):
    """
    读取 2D 图像并转换为模型所需的 Tensor 格式 [1, 1, H, W]
    """
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"无法读取图像: {img_path}")
    
    if is_mask:
        # 动态自适应阈值：以防掩码保存时像素值是 1 而不是 255
        max_val = img.max()
        if max_val == 0:
            print(f"\n[警告] 发现完全空白的掩码文件: {os.path.basename(img_path)} (这会导致该样本厚度误差算出来是0)")
            img = np.zeros_like(img, dtype=np.float32)
        else:
            # 只要像素值大于最大值的一半，就认为是有效掩码
            img = (img > (max_val / 2.0)).astype(np.float32)
    else:
        # 生成图归一化到 [0, 1]
        img = img.astype(np.float32) / 255.0

    # 转换为 Tensor，增加 Batch 和 Channel 维度 -> [1, 1, H, W]
    tensor = torch.from_numpy(img).unsqueeze(0).unsqueeze(0).to(device)
    return tensor

def evaluate_offline_thickness(results_dir, k=5, scale=1.0, device='cuda'):
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    thick_evaluator = ThicknessLoss(k=k, scale=scale).to(device)
    thick_evaluator.eval()

    # 直接寻找所有带有 _pred.png 后缀的预测图
    search_pattern = os.path.join(results_dir, '*_pred.png')
    pred_files = glob.glob(search_pattern)
    
    if not pred_files:
        print(f"在 {results_dir} 中没有找到任何 _pred.png 图像文件。")
        return

    errors = []

    with torch.no_grad():
        for pred_path in tqdm(pred_files, desc="Calculating Thickness Errors"):
            # 通过替换后缀，自动寻找对应的掩码图路径
            mask_path = pred_path.replace('_pred.png', '_mask.png')
            
            if not os.path.exists(mask_path):
                print(f"\n警告: 找不到对应的掩码文件 {mask_path}，已跳过。")
                continue

            # 读取张量
            pred_tensor = load_image_as_tensor(pred_path, is_mask=False, device=device)
            mask_tensor = load_image_as_tensor(mask_path, is_mask=True, device=device)

            # 计算单张图像的厚度误差
            error = thick_evaluator(pred_tensor, mask_tensor)
            errors.append(error.item())

    if errors:
        mean_error = np.mean(errors)
        std_error = np.std(errors)
        print("\n=== 厚度评估结果 ===")
        print(f"评估样本数: {len(errors)}")
        print(f"平均厚度误差 (MAE): {mean_error:.2f} um")
        print(f"厚度误差标准差 (Std): {std_error:.2f} um")
    else:
        print("未能成功计算任何有效样本的误差。")

if __name__ == '__main__':
    # ================= 配置区域 =================
    # 指向我们在 test_GAN.py 中保存所有图像的统一目录
    RESULTS_DIR = './results/saved_images'  
    
    # 物理分辨率缩放 (需要与原训练代码中的 args.scale_um 保持一致)
    SCALE_UM = 10.042947294612599 # 使用你在 test_GAN.py 中默认的 scale_um
    # 膨胀核大小 (需要与原训练代码中的 args.dilate_k 保持一致)
    DILATE_K = 1   
    # ==========================================
    
    evaluate_offline_thickness(
        results_dir=RESULTS_DIR, 
        k=DILATE_K, 
        scale=SCALE_UM, 
        device='cuda' # 如果没有 GPU，代码会自动回退到 'cpu'
    )