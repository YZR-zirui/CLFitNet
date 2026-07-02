import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


import argparse
import numpy as np
import math
import datetime
import torch.optim.lr_scheduler
from utils import *
from DataSet import Data_Center
import torch.nn as nn
from core import utils, trainer
import shutil
from Loss import ThicknessLoss # 确保在顶部导入
from tqdm import tqdm
from functions_GAN import train, test
import time
import torch.nn.functional as F
import torchvision.models as models
from torchvision.transforms import Normalize
from torch.nn import DataParallel
import torch.optim as optim
from Discriminator import *

parser = argparse.ArgumentParser(description='PredRNN - Pytorch')

# train&test
parser.add_argument('--device', type=str, default='cuda')

# data
parser.add_argument('--dataset_name', type=str, default='SCL')
parser.add_argument('--train_data_path', type=str, default='/data/yzr/CLFitNet_ICL_sobel/SCL')
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--img_width', type=int, default=512)
parser.add_argument('--total_length', type=int, default=5)


# ================= 动态厚度损失的物理标定 =================
# 图像真实尺寸: 深度 11.0mm, 对应像素 796, 组织折射率 1.376
# 将垂直方向的 mm/px 转换为 um/px
scale_um_val = (11.0 / 796.0 / 1.376) * 1000.0 

parser.add_argument('--lambda_thick', type=float, default=0.001, help='Weight for dynamic thickness loss')
parser.add_argument('--dilate_k', type=int, default=1, help='Kernel size for mask dilation in thickness loss')
parser.add_argument('--scale_um', type=float, default=scale_um_val, help='Physical scale um/pix (approx 10.048)')
# ================================================================


parser.add_argument('--res_dir', default='./results', type=str)
parser.add_argument('--model', default='SwinLSTM-M', type=str, help='Model type')
parser.add_argument('--patch_size', default=4, type=int, help='Patch size of input images')
parser.add_argument('--input_channels', default=1, type=int, help='Number of input image channels')
parser.add_argument('--embed_dim', default=160, type=int, help='Patch embedding dimension')
parser.add_argument('--depths', default=[12], type=int, help='Depth of Swin Transformer layer for SwinLSTM-B')
parser.add_argument('--depths_down', default=[2, 2, 2, 2], type=int, help='Downsample of SwinLSTM-D')
parser.add_argument('--depths_up', default=[2, 2, 2, 2], type=int, help='Upsample of SwinLSTM-D')
parser.add_argument('--window_size', default=8, type=int, help='Window size of Swin Transformer layer')
parser.add_argument('--drop_rate', default=0.5, type=float, help='Dropout rate')
parser.add_argument('--attn_drop_rate', default=0.05, type=float, help='Attention dropout rate')
parser.add_argument('--drop_path_rate', default=0.1, type=float, help='Stochastic depth rate')

parser.add_argument('--log_train', default=1, type=int)
parser.add_argument('--log_test', default=1, type=int)
parser.add_argument('--heads_number', default=[4, 8, 16, 16], type=int,
                    help='Number of attention heads in different layers')
# train
parser.add_argument('--seed', default=1234, type=int)
parser.add_argument('--test_batch_size', type=int, default=1)
# optimization
parser.add_argument('--lr', type=float, default=0.0001)
parser.add_argument('--epochs', type=int, default=400)
parser.add_argument('--gradient_accumulation_steps', type=int, default=4) 

args = parser.parse_args()
print(args)


def setup(args):
    if args.model == 'SwinLSTM-D':
        from SwinLSTM_D import SwinLSTM
        model = SwinLSTM(img_size=args.img_width, patch_size=args.patch_size,
                         in_chans=args.input_channels, embed_dim=args.embed_dim,
                         depths_downsample=args.depths_down, depths_upsample=args.depths_up,
                         num_heads=args.heads_number, window_size=args.window_size)
        model.to(args.device)

    if args.model == 'SwinLSTM-M':
        from SwinLSTM_MU_new import SwinLSTM
        model = SwinLSTM(img_size=args.img_width, patch_size=args.patch_size,
                         in_chans=args.input_channels, embed_dim=args.embed_dim,
                         depths_downsample=args.depths_down, depths_upsample=args.depths_up,
                         num_heads=args.heads_number, window_size=args.window_size)

        D1 = PatchDiscriminator(in_channels=1).to(args.device)
        D2 = PatchDiscriminator(in_channels=1).to(args.device)
        
        # 加载预训练权重
        # state_dict1 = torch.load('/data/yzr/CLFitNet_ICL/results/model/Swin_GAN_Gnew_70_best')
        # state_dict2 = torch.load('/data/yzr/CLFitNet_ICL/results/model/Swin_GAN_D1_70_best')
        # state_dict3 = torch.load('/data/yzr/CLFitNet_ICL/results/model/Swin_GAN_D2_70_best')
        # model.load_state_dict(state_dict1)
        # D1.load_state_dict(state_dict2)
        # D2.load_state_dict(state_dict3)
        
        model.to(args.device)

    optim_G = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.5, 0.999))
    optim_D1 = torch.optim.Adam(D1.parameters(), lr=0.00001, betas=(0.9, 0.999))
    optim_D2 = torch.optim.Adam(D2.parameters(), lr=0.00001, betas=(0.9, 0.999))
    
    D_criterion = nn.BCELoss()
    G_criterion = nn.MSELoss()
    
    sche_D1 = optim.lr_scheduler.StepLR(optim_D1, step_size=500, gamma=0.1)
    sche_D2 = optim.lr_scheduler.StepLR(optim_D2, step_size=500, gamma=0.1)
    sche_G = optim.lr_scheduler.StepLR(optim_G, step_size=200, gamma=0.1)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=150, gamma=0.1)
    
    mseloss = nn.MSELoss()
    perceptual_loss = 0
    
    # 实例化厚度损失函数
    thick_criterion = ThicknessLoss(k=args.dilate_k, scale=args.scale_um).to(args.device)

    return model, perceptual_loss, mseloss, optimizer, scheduler, optim_D1, optim_D2, optim_G, D_criterion, G_criterion, D1, D2, sche_D1, sche_D2, sche_G, thick_criterion


def main():
    set_seed(args.seed)
    train_input_handle = Data_Center.data_provider(args.dataset_name, args.train_data_path,
                                                   args.train_data_path, args.batch_size,
                                                   args.img_width,
                                                   seq_length=args.total_length, mode='train')
    valid_input_handle = Data_Center.data_provider(args.dataset_name, args.train_data_path,
                                                   args.train_data_path, args.batch_size,
                                                   args.img_width,
                                                   seq_length=args.total_length, mode='valid')
                                                   
    # 接收 thick_criterion
    model, perceptual_loss, mseloss, optimizer, scheduler, optim_D1, optim_D2, optim_G, D_criterion, G_criterion, D1, D2, sche_D1, sche_D2, sche_G, thick_criterion = setup(args)
    
    cache_dir, model_dir, log_dir = make_dir(args)
    logger = init_logger(log_dir)
    best_val_loss = float('inf')
    
    for epoch in range(args.epochs):
        start_time = time.time()
        print(f'-----------------------------Epoch {epoch+1}/{args.epochs} - Train-----------------------------')
        # 调用 train 时传入 thick_criterion
        train_loss = train(args, logger, epoch, model, train_input_handle, perceptual_loss, 
                           mseloss, optimizer, optim_D1, optim_D2, optim_G, D_criterion,
                           G_criterion, D1, D2, thick_criterion)

        current_lr = optimizer.param_groups[0]['lr']
        print(f'Time: {time.time() - start_time:.0f}s ; trainloss: {train_loss} ; lr: {current_lr}')
        logger.info(f'EP:{(int(epoch)):04d} Loss:{train_loss:.6f}')

        sche_D1.step()
        sche_D2.step()
        sche_G.step()

        # 验证模型
        interval = 5
        start_time1 = time.time()
        valid_loss, _, _, _, _ = test(args, logger, 0, model, valid_input_handle, mseloss, cache_dir)
        print(f'Time: {time.time() - start_time1:.0f}s ; validloss: {valid_loss}')

        if valid_loss < best_val_loss:
            best_val_loss = valid_loss
            best_model_state = model.state_dict()
            best_D1 = D1.state_dict()
            best_D2 = D2.state_dict()
            best_epoch = epoch
            
        if epoch % interval == 0 and epoch != 0:
            torch.save(best_model_state, f'{model_dir}/Swin_GAN_Gnew_{int(best_epoch)}_best')
            torch.save(best_D1, f'{model_dir}/Swin_GAN_D1_{int(best_epoch)}_best')
            torch.save(best_D2, f'{model_dir}/Swin_GAN_D2_{int(best_epoch)}_best')
            print(f'[Best_Model every {interval} epoch]  Loss:{best_val_loss:.4f}')
            best_val_loss = float('inf')
            best_model_state = None

if __name__ == '__main__':
    print('Initializing models for training...')
    main()

