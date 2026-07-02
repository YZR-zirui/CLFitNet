import numpy as np
import torch
from torch.cuda import amp
from torch.cuda.amp import autocast as autocast
from utils import compute_metrics, visualize
from tqdm import tqdm
from dataAug import apply_augmentations
from Discriminator import *
from Loss import *

scaler = torch.amp.GradScaler('cuda')

import torch.nn.functional as F

def model_forward_single_layer(model, inputs, targets_len, num_layers):
    outputs = []
    states = [None] * len(num_layers)
    inputs_len = inputs.shape[1]
    last_input = inputs[:, -1]
    for i in range(inputs_len - 1):
        output, states = model(inputs[:, i], states)
        outputs.append(output)
    for i in range(targets_len):
        output, states = model(last_input, states)
        outputs.append(output)
        last_input = output
    return outputs

def model_forward_multi_layer(model, inputs, targets_len, num_layers):
    states_down = [None] * len(num_layers)
    states_up = [None] * len(num_layers)
    outputs = []
    inputs_len = inputs.shape[1]
    last_input = inputs[:, -1]
    for i in range(inputs_len - 1): 
        output, states_down, states_up = model(inputs[:, i], states_down, states_up)
        outputs.append(output)
    for i in range(targets_len): 
        output, states_down, states_up = model(last_input, states_down, states_up)
        outputs.append(output)
        last_input = output
    return outputs

def model_forward_multi_layer_U(model, inputs, targets_len, num_layers):
    states_down = [None] * len(num_layers)
    states_up = [None] * len(num_layers)
    outputs = []
    inputs_len = inputs.shape[1]
    last_input = inputs[:, -1]
    for i in range(inputs_len - 1):
        output, states_down, states_up = model(inputs[:, i], states_down, states_up)
    for i in range(targets_len): 
        output, states_down, states_up = model(last_input, states_down, states_up)
        outputs.append(output) 
        last_input = output
    return outputs


# 修改 train 函数签名
def train(args, logger, epoch, model, train_input_handle, perceptual_loss, mseloss, optimizer, optim_D1, optim_D2, optim_G, D_criterion, G_criterion, D1, D2, thick_criterion):
    perloss = GPatchGANLosses(args.device)
    ssimloss = SSIMLoss()
    model.train()
    num_batches = train_input_handle.total_batch()
    L2loss = torch.nn.MSELoss()
    pbar = tqdm(range(num_batches))
    losses = []
    for batch_idx in pbar:
        if train_input_handle.no_batch_left():
            train_input_handle.begin(do_shuffle=True)
            
        ims = train_input_handle.get_batch()
        
        inputs = ims[:, 0:3, :, :, :]  
        targets = ims[:, 3:5, :, :, :] 
        masks = ims[:, 5:7, :, :, :]    # [B, 2, H, W, 1] 新增：获取目标掩码
        
        inputs = torch.FloatTensor(inputs).to(args.device)
        targets = torch.FloatTensor(targets).to(args.device)
        masks = torch.FloatTensor(masks).to(args.device) # 新增：转移到GPU

        inputs = inputs.permute(0, 1, 4, 2, 3).contiguous()
        targets = targets.permute(0, 1, 4, 2, 3).contiguous() 
        masks = masks.permute(0, 1, 4, 2, 3).contiguous() # 新增：调整维度 [B, 2, 1, H, W]
        targets_len = targets.shape[1]

        fake_images = model_forward_multi_layer_U(model, inputs, targets_len, args.depths_down) 
        
        fake_images = torch.stack(fake_images).permute(1, 0, 2, 3, 4).contiguous()
        fake_image1 = fake_images[:,0]
        fake_image2 = fake_images[:,1]
        real_image1 = targets[:,0]
        real_image2 = targets[:,1]
        
        # 取出对应的两帧 Mask
        mask1 = masks[:,0]
        mask2 = masks[:,1]

        # 训练判别器
        M = 1
        for _ in range(M):
            optim_D1.zero_grad()
            real_output1 = D1(real_image1) 
            fake_output1 = D1(fake_image1.detach()) # 判别器训练时，这里必须用 detach()
            real_labels1 = torch.ones_like(real_output1) 
            fake_labels1 = torch.zeros_like(fake_output1)
            D1_total_loss = discriminator_loss(real_output1, fake_output1, real_labels1, fake_labels1)
            D1_total_loss.backward()
            torch.nn.utils.clip_grad_norm_(D1.parameters(), max_norm=1.0)
            optim_D1.step()
            
            optim_D2.zero_grad()
            real_output2 = D2(real_image2)
            fake_output2 = D2(fake_image2.detach())
            real_labels2 = torch.ones_like(real_output2)
            fake_labels2 = torch.zeros_like(fake_output2)
            D2_total_loss = discriminator_loss(real_output2, fake_output2, real_labels2, fake_labels2)
            D2_total_loss.backward()
            torch.nn.utils.clip_grad_norm_(D2.parameters(), max_norm=1.0)
            optim_D2.step()
            
            d1_loss_val = D1_total_loss.item()
            d2_loss_val = D2_total_loss.item()

        # 训练生成器
        optim_G.zero_grad()
        real_output1 = D1(real_image1)
        fake_output1 = D1(fake_image1) #fake_output1 = D1(fake_image1.detach())
        real_output2 = D2(real_image2)
        fake_output2 = D2(fake_image2)#fake_output2 = D2(fake_image2.detach())
        
        loss_per1 = perloss.perceptual_loss(fake_image1, real_image1)
        loss_per2 = perloss.perceptual_loss(fake_image2, real_image2)
        loss_ssim1 = ssimloss(fake_image1, real_image1)
        loss_ssim2 = ssimloss(fake_image2, real_image2)
        L2loss1 = L2loss(fake_image1, real_image1)
        L2loss2 = L2loss(fake_image2, real_image2)
        sobelloss1 = sobel_loss(fake_image1, real_image1)
        sobelloss2 = sobel_loss(fake_image2, real_image2)
        
        # === 新增：计算动态厚度损失 ===
        loss_thick1 = thick_criterion(fake_image1, mask1)
        loss_thick2 = thick_criterion(fake_image2, mask2)
        loss_thick = (0.5 * loss_thick1 + 0.5 * loss_thick2)
        
        g_advloss_ = g_advloss(fake_output1,fake_output2,target_value=1.0)
        
        loss_per = (0.5*loss_per1 + 0.5*loss_per2)
        loss_ssim = (0.5*loss_ssim1 + 0.5*loss_ssim2)
        L2loss_ = (0.5*L2loss1 + 0.5*L2loss2)
        # sobelloss = (0.5*sobelloss1 + 0.5*sobelloss2)

        # ssim + thick
        # g_loss = g_advloss_ + loss_ssim + (args.lambda_thick * loss_thick)

        # 汇总 G_loss（可以根据情况调节 lambda_thick 权重，这里默认用 args.lambda_thick）
        g_loss = g_advloss_ + loss_per + loss_ssim + (args.lambda_thick * loss_thick)
  
        g_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optim_G.step()

        losses.append(g_loss.item())
        pbar.set_postfix({
            'G_avg': f'{np.mean(losses):.4f}',
            'D1': f'{d1_loss_val:.4f}',
            'D2': f'{d2_loss_val:.4f}'
        })
        train_input_handle.next()

    return np.mean(losses)

# def test(args, logger, epoch, model, test_input_handle, criterion, cache_dir):
#     model.eval()
#     num_batches = test_input_handle.total_batch()
#     losses, mses, ssims = [], [], []
#     for batch_idx in tqdm(range(num_batches)):
#         if test_input_handle.no_batch_left():
#             test_input_handle.begin(do_shuffle=False)
            
#         # 测试阶段，忽略 xL, xR, CSJ
#         ims, _, _, _ = test_input_handle.get_batch()
        
#         inputs = ims[:, 0:3, :, :, :] 
#         targets = ims[:, 3:5, :, :, :] 
#         inputs = torch.FloatTensor(inputs).to(args.device)
#         targets = torch.FloatTensor(targets).to(args.device)

#         inputs = inputs.permute(0, 1, 4, 2, 3).contiguous()
#         targets = targets.permute(0, 1, 4, 2, 3).contiguous()

#         with torch.no_grad():
#             targets_len = targets.shape[1]
#             outputs = model_forward_multi_layer_U(model, inputs, targets_len, args.depths_down)
#             outputs = torch.stack(outputs).permute(1, 0, 2, 3, 4).contiguous()
#             losses.append(criterion(outputs, targets).item())

#         test_input_handle.next()

#     return np.mean(losses)

# def test(args, logger, epoch, model, test_input_handle, criterion, cache_dir):
#     model.eval()
#     num_batches = test_input_handle.total_batch()
#     losses = []  # 保持你原本的命名和逻辑
    
#     # 为了满足 test_GAN 的指标需求，建立临时列表
#     all_preds = []
#     all_targets = []

#     # 确保数据句柄重置
#     test_input_handle.begin(do_shuffle=False)

#     for batch_idx in tqdm(range(num_batches), desc="Testing/Validating"):
#         if test_input_handle.no_batch_left():
#             break
            
#         ims = test_input_handle.get_batch()
        
#         inputs = ims[:, 0:3, :, :, :] 
#         targets = ims[:, 3:5, :, :, :] 
#         inputs = torch.FloatTensor(inputs).to(args.device)
#         targets = torch.FloatTensor(targets).to(args.device)

#         inputs = inputs.permute(0, 1, 4, 2, 3).contiguous()
#         targets = targets.permute(0, 1, 4, 2, 3).contiguous()

#         with torch.no_grad():
#             targets_len = targets.shape[1]
#             outputs = model_forward_multi_layer_U(model, inputs, targets_len, args.depths_down)
#             outputs = torch.stack(outputs).permute(1, 0, 2, 3, 4).contiguous()
            
#             # --- 原本的 Loss 计算逻辑 ---
#             loss_val = criterion(outputs, targets).item()
#             losses.append(loss_val)
#             # --------------------------

#             # 仅为了满足 test_GAN 的多指标返回，收集数据
#             all_preds.append(outputs.cpu())
#             all_targets.append(targets.cpu())

#         test_input_handle.next()

#     # 1. 计算原本的平均 Loss (用于验证)
#     avg_loss = np.mean(losses)

#     # 2. 计算额外的测试指标 (用于测试)
#     all_preds = torch.cat(all_preds, dim=0)
#     all_targets = torch.cat(all_targets, dim=0)
#     mse, ssim, psnr = compute_metrics(all_preds, all_targets)
#     rmse = np.sqrt(mse)

#     # 3. 统一返回 5 个值
#     # 第一个值 avg_loss 就是你原本的 np.mean(losses)
#     return avg_loss, mse, ssim, rmse, psnr

def test(args, logger, epoch, model, test_input_handle, criterion, cache_dir):
    model.eval()
    num_batches = test_input_handle.total_batch()
    losses = []  
    
    # 使用列表存储每个 batch/样本 的单独指标
    thick_errors = [] 
    mses = []
    psnrs = []
    rmses = []

    # 初始化厚度计算器
    thick_evaluator = ThicknessLoss(k=args.dilate_k, scale=args.scale_um).to(args.device)

    # 确保数据句柄重置
    test_input_handle.begin(do_shuffle=False)

    for batch_idx in tqdm(range(num_batches), desc="Testing/Validating"):
        if test_input_handle.no_batch_left():
            break
            
        batch_data = test_input_handle.get_batch()
        # 如果是元组，说明包含文件名，我们只取第一个元素（图像数据）
        if isinstance(batch_data, tuple):
            ims = batch_data[0]
        else:
            ims = batch_data
        
        inputs = ims[:, 0:3, :, :, :] 
        targets = ims[:, 3:5, :, :, :] 
        masks = ims[:, 5:7, :, :, :] 
        
        inputs = torch.FloatTensor(inputs).to(args.device)
        targets = torch.FloatTensor(targets).to(args.device)
        masks = torch.FloatTensor(masks).to(args.device) 

        inputs = inputs.permute(0, 1, 4, 2, 3).contiguous()
        targets = targets.permute(0, 1, 4, 2, 3).contiguous()
        masks = masks.permute(0, 1, 4, 2, 3).contiguous() 

        with torch.no_grad():
            targets_len = targets.shape[1]
            outputs = model_forward_multi_layer_U(model, inputs, targets_len, args.depths_down)
            outputs = torch.stack(outputs).permute(1, 0, 2, 3, 4).contiguous()
            
            # --- Loss 计算 ---
            loss_val = criterion(outputs, targets).item()
            losses.append(loss_val)
            
            # --- 计算物理厚度绝对误差 (MAE in um) ---
            thick_err1 = thick_evaluator(outputs[:, 0], masks[:, 0])
            thick_err2 = thick_evaluator(outputs[:, 1], masks[:, 1])
            batch_thick_mae = (thick_err1 + thick_err2) / 2.0
            thick_errors.append(batch_thick_mae.item())

            # --- 计算该 Batch 的其他指标 ---
            # 直接在这个循环里调用 compute_metrics
            batch_mse, _, batch_psnr = compute_metrics(outputs.cpu(), targets.cpu())
            mses.append(batch_mse)
            psnrs.append(batch_psnr)
            rmses.append(np.sqrt(batch_mse))

        test_input_handle.next()

    # 1. 计算平均 Loss
    avg_loss = np.mean(losses)

    # 2. 计算各个指标的 均值(mean) 和 标准差(std)
    # 格式打包为 (mean, std)
    mse_stats = (np.mean(mses), np.std(mses))
    thick_stats = (np.mean(thick_errors), np.std(thick_errors))
    rmse_stats = (np.mean(rmses), np.std(rmses))
    psnr_stats = (np.mean(psnrs), np.std(psnrs))

    # 3. 统一返回
    return avg_loss, mse_stats, thick_stats, rmse_stats, psnr_stats
