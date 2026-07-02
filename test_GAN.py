import argparse
import numpy as np
import math
import datetime
import torch
import torch.nn as nn
from utils import *
from DataSet import Data_Center
from core import utils, trainer
import os
import cv2  # 新增：用于保存图片
from tqdm import tqdm
from functions_GAN import test, model_forward_multi_layer_U # 新增导入模型前向函数
import time
from Discriminator import *
import warnings
warnings.filterwarnings("ignore", message="You are using `torch.load` with `weights_only=False`")

parser = argparse.ArgumentParser(description='PredRNN - Pytorch - Test Only')

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

# train&test
parser.add_argument('--device', type=str, default='cuda')

# 动态厚度损失参数
parser.add_argument('--dilate_k', type=int, default=1, help='Kernel size for mask dilation in thickness loss')
parser.add_argument('--scale_um', type=float, default=10.042947294612599, help='Physical scale um/pix')

# data
parser.add_argument('--dataset_name', type=str, default='SCL')
parser.add_argument('--train_data_path', type=str, default='/data/yzr/CLFitNet_ICL_sobel/SCL')
parser.add_argument('--test_batch_size', type=int, default=1)
parser.add_argument('--img_width', type=int, default=512)
parser.add_argument('--total_length', type=int, default=5)

parser.add_argument('--res_dir', default='./results', type=str)
parser.add_argument('--save_dir', default='./results/saved_images', type=str, help='Dir to save generated images') # 新增保存路径
parser.add_argument('--model', default='SwinLSTM-M', type=str, help='Model type')
parser.add_argument('--patch_size', default=4, type=int, help='Patch size of input images')
parser.add_argument('--input_channels', default=1, type=int, help='Number of input image channels')
parser.add_argument('--embed_dim', default=160, type=int, help='Patch embedding dimension')
parser.add_argument('--depths_down', default=[2, 2, 2, 2], type=int, help='Downsample of SwinLSTM-D')
parser.add_argument('--depths_up', default=[2, 2, 2, 2], type=int, help='Upsample of SwinLSTM-D')
parser.add_argument('--window_size', default=8, type=int, help='Window size of Swin Transformer layer')
parser.add_argument('--heads_number', default=[4, 8, 16, 16], type=int, help='Number of attention heads')
parser.add_argument('--seed', default=1234, type=int)

args = parser.parse_args()
print(args)


def save_visualizations(args, model, test_input_handle):
    """
    独立的可视化保存函数：运行模型，并将生成图(pred)、真实图(gt)、掩码图(mask)一同保存。
    """
    print(f'-----------------------------Saving Images to {args.save_dir}-----------------------------')
    os.makedirs(args.save_dir, exist_ok=True)
    model.eval()
    
    # 重置数据句柄，准备从头开始遍历测试集
    test_input_handle.begin(do_shuffle=False)
    num_batches = test_input_handle.total_batch()
    
    # 如果不想保存测试集里所有的图，可以限制一个数量，例如只保存前 100 个 batch
    max_save_batches = 390
    save_count = 0

    with torch.no_grad():
        for batch_idx in tqdm(range(num_batches), desc="Saving Images"):
            if test_input_handle.no_batch_left() or save_count >= max_save_batches:
                break

            batch_data = test_input_handle.get_batch()
            
            # 安全解包（兼容是否返回文件名的两种情况）
            if isinstance(batch_data, tuple):
                ims = batch_data[0]
            else:
                ims = batch_data
            
            # 提取前3帧作为输入
            inputs = ims[:, 0:3, :, :, :] 
            
            inputs_tensor = torch.FloatTensor(inputs).to(args.device)
            inputs_tensor = inputs_tensor.permute(0, 1, 4, 2, 3).contiguous()

            # 生成后两帧
            targets_len = 2 
            outputs = model_forward_multi_layer_U(model, inputs_tensor, targets_len, args.depths_down)
            outputs = torch.stack(outputs).permute(1, 0, 2, 3, 4).contiguous()

            # 将 Tensor 转回 numpy 用于 OpenCV 保存
            preds_np = outputs.cpu().numpy() 

            batch_size = preds_np.shape[0]
            for b in range(batch_size):
                # 生成统一的序列前缀，例如 "seq0001"
                seq_base_name = f"seq{save_count:04d}"

                for t in range(targets_len): # 遍历后两帧 (t=0对应frame4, t=1对应frame5)
                    # 1. 提取预测图像 (Prediction)
                    pred_img = preds_np[b, t, 0, :, :]
                    
                    # 2. 提取真实图像 (Ground Truth)，在 ims 的第 3, 4 索引
                    gt_img = ims[b, 3+t, :, :, 0]
                    
                    # 3. 提取掩码图像 (Mask)，在 ims 的第 5, 6 索引 
                    # (由于之前 data_SCL.py 把 input_batch(5帧) 和 mask_batch(2帧) 拼接成了7帧)
                    if ims.shape[1] >= 7:
                        mask_img = ims[b, 5+t, :, :, 0]
                    else:
                        mask_img = np.zeros_like(gt_img) # 兜底逻辑

                    # 映射回 [0, 255] 像素值域
                    pred_img = np.clip(pred_img * 255.0, 0, 255).astype(np.uint8)
                    gt_img = np.clip(gt_img * 255.0, 0, 255).astype(np.uint8)
                    mask_img = np.clip(mask_img * 255.0, 0, 255).astype(np.uint8)

                    # 构建独立的文件名：使用相同的前缀，仅靠后缀区分
                    pred_filename = os.path.join(args.save_dir, f"{seq_base_name}_frame{t+4}_pred.png")
                    gt_filename   = os.path.join(args.save_dir, f"{seq_base_name}_frame{t+4}_gt.png")
                    mask_filename = os.path.join(args.save_dir, f"{seq_base_name}_frame{t+4}_mask.png")

                    # 分别保存三张图片
                    cv2.imwrite(pred_filename, pred_img)
                    cv2.imwrite(gt_filename, gt_img)
                    cv2.imwrite(mask_filename, mask_img)
                
                save_count += 1

            test_input_handle.next()
            
    print(f'Successfully saved {save_count} sequences (each containing pred, gt, mask) to {args.save_dir}')


def main():
    print('-----------------------------begin test-----------------------------')
    set_seed(args.seed)
    cache_dir, model_dir, log_dir = make_dir(args)
    logger = init_logger(log_dir)

    # 1. 动态按需导入模型
    if args.model == 'SwinLSTM-M':
        from SwinLSTM_MU_new import SwinLSTM
    elif args.model == 'SwinLSTM-D':
        from SwinLSTM_D import SwinLSTM
    else:
        raise ValueError(f"Unsupported model: {args.model}")

    # 2. 实例化模型
    model = SwinLSTM(img_size=args.img_width, patch_size=args.patch_size,
                     in_chans=args.input_channels, embed_dim=args.embed_dim,
                     depths_downsample=args.depths_down, depths_upsample=args.depths_up,
                     num_heads=args.heads_number, window_size=args.window_size).to(args.device)
                     
    D1 = PatchDiscriminator(in_channels=1).to(args.device)
    D2 = PatchDiscriminator(in_channels=1).to(args.device)

    # 3. 加载权重
    model_state_dict = torch.load('results/ssim_thick_model/Swin_GAN_Gnew_202_best')
    D1_state_dict = torch.load('results/ssim_thick_model/Swin_GAN_D1_202_best')
    D2_state_dict = torch.load('results/ssim_thick_model/Swin_GAN_D2_202_best')

    parallel = False 
    if parallel:
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in model_state_dict.items():
            name = k[7:] if k.startswith('module.') else k
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)
    else:
        model.load_state_dict(model_state_dict)
        
    D1.load_state_dict(D1_state_dict)
    D2.load_state_dict(D2_state_dict)

    # 4. 设置为评估模式
    model.eval()
    D1.eval()
    D2.eval()

    # 5. 加载测试数据
    test_input_handle = Data_Center.data_provider(args.dataset_name, args.train_data_path,
                                                  args.train_data_path, args.test_batch_size,
                                                  args.img_width,
                                                  seq_length=args.total_length, mode='test')
                                                  
    criterion = nn.MSELoss().to(args.device)

    start_time = time.time()

    # 6. 计算评估指标并打印
    # 现在接收到的是包含 (mean, std) 的元组
    _, mse_stats, thick_stats, rmse_stats, psnr_stats = test(args, logger, 0, model, test_input_handle, criterion, cache_dir)
    
    print(f'[Metrics]  '
          f'MSE: {mse_stats[0]:.4f}±{mse_stats[1]:.4f} | '
          f'Thick_MAE(um): {thick_stats[0]:.2f}±{thick_stats[1]:.2f} | '
          f'RMSE: {rmse_stats[0]:.4f}±{rmse_stats[1]:.4f} | '
          f'PSNR: {psnr_stats[0]:.2f}±{psnr_stats[1]:.2f}')
          
    print(f'Metrics calculation time usage: {time.time() - start_time:.0f}s')

    # 7. 执行图像生成并保存
    save_visualizations(args, model, test_input_handle)

if __name__ == '__main__':
    main()
