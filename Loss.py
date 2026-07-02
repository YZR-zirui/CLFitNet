from torchvision.models import VGG16_Weights
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import cv2
import numpy as np
from torch.autograd import Variable
# 定义VGGFeatureExtractor


def g_advloss(D1_outputs, D2_outputs, target_value=1.0):
    """
    计算生成器的对抗损失。
    判别器最后一层没有加 Sigmoid，直接输出了 Logits，
    因此这里使用 BCEWithLogitsLoss 来保证数值稳定并与判别器对齐。
    """
    criterion = nn.BCEWithLogitsLoss()
    
    # 生成器的目标是让判别器认为假图像是真的 (赋予 1.0 的标签)
    target1 = torch.full_like(D1_outputs, target_value)
    target2 = torch.full_like(D2_outputs, target_value)
    
    loss_D1 = criterion(D1_outputs, target1)
    loss_D2 = criterion(D2_outputs, target2)
    
    total_loss = (loss_D1 + loss_D2) / 2.0
    return total_loss


# def g_advloss(fakeoutput,realoutput):
#     real_loss = F.binary_cross_entropy_with_logits(realoutput - fakeoutput, torch.ones_like(realoutput))
#     fake_loss = F.binary_cross_entropy_with_logits(fakeoutput - realoutput, torch.zeros_like(fakeoutput))
#     return (real_loss + fake_loss) / 2

# def g_advloss(D1_outputs, D2_outputs, target_value=1.0):
#     """
#     计算对抗损失，基于鉴别器输出和指定目标值之间的F范数平方差。
#     :param D_M_outputs: 来自第一个鉴别器的输出，形状为 [batch_size, 1, 16, 16]
#     :param D_P_outputs: 来自第二个鉴别器的输出，形状为 [batch_size, 1, 16, 16]
#     :param target_value: 鉴别器的输出目标值
#     :return: 对抗损失
#     """
#     N = D1_outputs.size(0)
#     ideal_output = torch.full_like(D1_outputs, target_value)

#     loss_D1 = torch.norm(D1_outputs - ideal_output, p='fro')
#     loss_D2 = torch.norm(D2_outputs - ideal_output, p='fro')

#     total_loss = (loss_D1 + loss_D2) / N

#     return total_loss

# def g_advloss(fakeoutput,realoutput):
#     # 生成器的目标是让判别器认为生成的图像是真实的，即尽可能接近1
#     target = torch.ones_like(fakeoutput)
#     loss = torch.mean((fakeoutput - target) ** 2)
#     return loss


import torch
def sobel_edges(img):
    """
    使用Sobel算子计算图像的边缘。
    :param img: 输入图像，tensor格式，[B, C, H, W]
    :return: Sobel边缘图，同样大小，[B, C, H, W]
    """
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view((1, 1, 3, 3)).to(img.device)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view((1, 1, 3, 3)).to(img.device)

    img_gray = img.mean(dim=1, keepdim=True)  # 转为灰度图
    edges_x = F.conv2d(img_gray, sobel_x, padding=1)
    edges_y = F.conv2d(img_gray, sobel_y, padding=1)
    edges = torch.sqrt(edges_x ** 2 + edges_y ** 2)

    return edges

def sobel_loss(pred, target):
    """
    计算预测图像与目标图像之间的Sobel边缘损失。
    :param pred: 预测图像，tensor格式，[B, C, H, W]
    :param target: 目标图像，tensor格式，[B, C, H, W]
    """
    pred_edges = sobel_edges(pred)
    target_edges = sobel_edges(target)
    return F.mse_loss(pred_edges, target_edges)


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
        # 在 PyTorch 中，使用 max_pool2d 步长为 1 可以完美等效于形态学膨胀
        M_ROI = F.max_pool2d(mask, kernel_size=self.k, stride=1, padding=self.k//2)
        
        # 3. 提取目标梯度
        G_focus = grad_y * M_ROI
        
        # 4. 分离上下边
        E_up = F.relu(G_focus)
        E_down = F.relu(-G_focus)
        
        # 5. 计算垂直质心 (Soft-argmax)
        # 生成 Y 坐标矩阵 [1, 1, H, 1]
        y_coords = torch.arange(H, device=pred.device, dtype=torch.float32).view(1, 1, H, 1)
        
        sum_E_up = torch.sum(E_up, dim=2, keepdim=True) + eps
        sum_E_down = torch.sum(E_down, dim=2, keepdim=True) + eps
        
        y_bar_up = torch.sum(y_coords * E_up, dim=2, keepdim=True) / sum_E_up
        y_bar_down = torch.sum(y_coords * E_down, dim=2, keepdim=True) / sum_E_down
        
        # 6. 计算软厚度
        T_soft = y_bar_down - y_bar_up  # 预测的软厚度
        
        # 补充：计算真实厚度 T_gt
        # 对于二值掩码，某一列的厚度就是该列值为1的像素总和
        T_gt = torch.sum(mask, dim=2, keepdim=True) 
        
        # 7. 转换成临床物理单位 (um)
        T_um_pred = T_soft * self.scale
        T_um_gt = T_gt * self.scale
        
        # 8. 最后，计算 L_thick (这里使用 L1 绝对值误差，且只在掩码存在的区域计算)
        valid_cols = (T_gt > 0).float() # 只计算有泪液层存在的列
        loss = torch.abs(T_um_pred - T_um_gt) * valid_cols
        
        # 取所有有效列的平均值
        L_thick = torch.sum(loss) / (torch.sum(valid_cols) + eps)
        
        return L_thick


#F范式平方约束
def Fpow(output, target):
    """
    计算输出和目标图像之间的F范数的平方损失。适用于形状为 [batch, 1, 512, 512] 的输入。
    Args:
        output (torch.Tensor): 模型的输出图像，形状为 [batch, 1, 512, 512]
        target (torch.Tensor): 目标图像，形状为 [batch, 1, 512, 512]
    Returns:
        torch.Tensor: 计算得到的平均损失值
    """
    # 计算两者之间的差异的平方
    difference = output - target
    squared_difference = difference ** 2
    # 计算F范数的平方，即差异的平方和
    # 因为是单通道图像，可以直接在通道、高度、宽度上求和
    norm_squared = squared_difference.sum(dim=[1, 2, 3])
    
    # 计算损失的平均值
    loss = torch.mean(norm_squared)
    return loss

    
class VGGFeatureExtractor(nn.Module):
    def __init__(self, device):
        super(VGGFeatureExtractor, self).__init__()
        vgg = models.vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features[:36]
        self.features = nn.Sequential(
            nn.Conv2d(1, 3, kernel_size=1),  # 将单通道扩展为三通道
            *vgg
        )
        self.features.eval()  # 设置为评估模式，固定VGG层的参数
        self.features.to(device)

    def forward(self, x):
        return self.features(x)  # 直接返回前36层的输出，这里不再单独处理每层的输出

# class VGGFeatureExtractor(nn.Module):
#     def __init__(self, device):
#         super(VGGFeatureExtractor, self).__init__()
#         # 加载预训练的VGG-16模型
#         vgg = models.vgg16(pretrained=True).features
#         # 提取直到block3_conv3的层
#         self.conv1_to_block3_conv3 = nn.Sequential(
#             nn.Conv2d(1, 3, kernel_size=1),  # 将单通道扩展为三通道
#             *vgg[:17]  # 包括block3_conv3（索引为16）
#         )
#         self.conv1_to_block3_conv3.eval()  # 设置为评估模式，固定参数
#         self.conv1_to_block3_conv3.to(device)

#     def forward(self, x):
#         # 仅执行到block3_conv3
#         block3_conv3_output = self.conv1_to_block3_conv3(x)
#         return block3_conv3_output

# 定义G-loss损失计算
class GPatchGANLosses:
    def __init__(self, device='cuda'):
        self.criterion = nn.MSELoss()
        self.vgg = VGGFeatureExtractor(device).to(device)

    def perceptual_loss(self, Image_fake, Image_real):
        phi_fake = self.vgg(Image_fake)
        phi_real = self.vgg(Image_real)
        return self.criterion(phi_fake, phi_real)  # 计算感知损失

# 定义像素级别损失
# Fun Loss
class FunLoss(nn.Module):
    def __init__(self):
        super(FunLoss, self).__init__()

    def forward(self, x, y):
        return torch.norm(x - y, p='fro') ** 2 / (x.shape[0] * x.shape[1] * x.shape[2] * x.shape[3])
# Grad Loss (Total Variation Loss)
class GradLoss(nn.Module):
    def __init__(self):
        super(GradLoss, self).__init__()

    def forward(self, x, y):
        def tv_loss(img):
            batch_size, c, h, w = img.size()
            h_tv = torch.pow(img[:, :, 1:, :] - img[:, :, :-1, :], 2).sum()
            w_tv = torch.pow(img[:, :, :, 1:] - img[:, :, :, :-1], 2).sum()
            return (h_tv + w_tv) / (batch_size * c * h * w)

        return tv_loss(x - y)


# SSIM Loss
class SSIMLoss(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True):
        super(SSIMLoss, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = self.create_window(window_size, self.channel)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = self.create_window(self.window_size, channel).to(img1.device)
            self.window = window
            self.channel = channel

        ssim_value = self.ssim(img1, img2, window=window, window_size=self.window_size, channel=channel, size_average=self.size_average)
        ssim_loss = (1 - ssim_value).clamp(min=0, max=1)  # Clamping to ensure loss is within 0 to 1
        return ssim_loss

    def ssim(self, img1, img2, window, window_size, channel, size_average=True):
        mu1 = F.conv2d(img1.clone(), window, padding=window_size // 2, groups=channel)
        mu2 = F.conv2d(img2.clone(), window, padding=window_size // 2, groups=channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(img1.clone() * img1.clone(), window, padding=window_size // 2, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(img2.clone() * img2.clone(), window, padding=window_size // 2, groups=channel) - mu2_sq
        sigma12 = F.conv2d(img1.clone()*img2.clone(), window, padding=window_size // 2, groups=channel) - mu1_mu2
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

        if size_average:
            return ssim_map.mean()
        else:
            return ssim_map.mean(1).mean(1).mean(1)

    def create_window(self, window_size, channel):
        def gaussian(window_size, sigma):
            gauss = torch.Tensor(
                [np.exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
            return gauss / gauss.sum()

        _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
        return window
    
def laplacian(image):
    # 定义拉普拉斯核
    laplacian_kernel = torch.tensor([[1, 1, 1],
                                     [1, -8, 1],
                                     [1, 1, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    
    # 确保在使用自定义卷积核时使用合适的padding保持尺寸不变
    laplacian_kernel = laplacian_kernel.to(image.device)
    padding = 1  # 核大小为3，因此padding设置为1
    
    # 卷积运算
    lap_response = F.conv2d(image, laplacian_kernel, padding=padding)
    return lap_response

def laplacian_loss(pred, target):
    # 计算预测和目标的拉普拉斯响应
    lap_pred = laplacian(pred)
    lap_target = laplacian(target)
    
    # 计算损失
    loss = F.mse_loss(lap_pred, lap_target)
    return loss


if __name__ == '__main__':
    device = torch.device('cuda')
    vgg_loss_calculator = GPatchGANLosses(device)
    Image_fake = torch.randn([5, 1, 512, 512], device=device)  # 假设的生成图像
    Image_real = torch.randn([5, 1, 512, 512], device=device)  # 假设的真实图像
    loss = vgg_loss_calculator.perceptual_loss(Image_fake, Image_real)
    print(f"Perceptual loss: {loss.item()}")
