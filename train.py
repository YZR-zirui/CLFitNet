import argparse
import numpy as np
import math
import datetime
import torch.optim.lr_scheduler
from utils import *
from DataSet import Data_Center
import torch.nn as nn
from core import utils, trainer
import os
import shutil
# from models import ModelCenter
from tqdm import tqdm
from functions import train,test
import time
import torch.nn.functional as F
import torchvision.models as models
from torchvision.transforms import Normalize
from torch.nn import DataParallel
import torch.optim as optim
from functions import model_forward_multi_layer_U,model_forward_multi_layer


parser = argparse.ArgumentParser(description='PredRNN - Pytorch')

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

# train&test
parser.add_argument('--device', type=str, default='cuda')

# data
parser.add_argument('--dataset_name', type=str, default='SCL')
parser.add_argument('--train_data_path', type=str, default=r"/data/yzr/nas/home/SCL")
parser.add_argument('--batch_size', type=int, default=1)    # 4
parser.add_argument('--img_width', type=int, default=256)   # 512
parser.add_argument('--total_length', type=int, default=5)

parser.add_argument('--res_dir', default='./results', type=str)
parser.add_argument('--model', default='SwinLSTM-D', type=str, choices=['SwinLSTM-U', 'SwinLSTM-D', 'SwinLSTM-M'], help='Model type')
parser.add_argument('--patch_size', default=2, type=int, help='Patch size of input images')
parser.add_argument('--input_channels', default=1, type=int, help='Number of input image channels')
parser.add_argument('--embed_dim', default=128, type=int, help='Patch embedding dimension')
parser.add_argument('--depths', default=[12], type=int, help='Depth of Swin Transformer layer for SwinLSTM-B')
parser.add_argument('--depths_down', default=[2,2,4,4], type=int, help='Downsample of SwinLSTM-D')
parser.add_argument('--depths_up', default=[4,4,2,2], type=int, help='Upsample of SwinLSTM-D')
parser.add_argument('--window_size', default=8, type=int, help='Window size of Swin Transformer layer')
# 添加dropout避免参数过大
parser.add_argument('--drop_rate', default=0.2, type=float, help='Dropout rate')
parser.add_argument('--attn_drop_rate', default=0.2, type=float, help='Attention dropout rate')
parser.add_argument('--drop_path_rate', default=0.1, type=float, help='Stochastic depth rate')

parser.add_argument('--log_train', default=1, type=int)
parser.add_argument('--log_test', default=1, type=int)
parser.add_argument('--heads_number', default=[4,4,8,8], type=int, help='Number of attention heads in different layers')
# train
parser.add_argument('--seed', default=1234, type=int)
parser.add_argument('--test_batch_size', type=int, default=1)
# optimization
parser.add_argument('--lr', type=float, default=0.0001)
parser.add_argument('--epochs',type=int,default=5)

args = parser.parse_args()
print(args)
    
def setup(args):
    """
    if args.model == 'SwinLSTM-B':
        from SwinLSTM_B import SwinLSTM
        model = SwinLSTM(img_size=args.img_width, patch_size=args.patch_size,
                         in_chans=args.input_channels, embed_dim=args.embed_dim,
                         depths=args.depths, num_heads=args.heads_number,
                         window_size=args.window_size, drop_rate=args.drop_rate,
                         attn_drop_rate=args.attn_drop_rate, drop_path_rate=args.drop_path_rate)
        model = nn.DataParallel(model,device_ids = [0,1,2,3,4,5,6,7])
        model.to(args.device)
    """
    if args.model == 'SwinLSTM-D':
        from SwinLSTM_D import SwinLSTM
        model = SwinLSTM(img_size=args.img_width, patch_size=args.patch_size,
                         in_chans=args.input_channels, embed_dim=args.embed_dim,
                         depths_downsample=args.depths_down, depths_upsample=args.depths_up,
                         num_heads=args.heads_number, window_size=args.window_size)
        model.to(args.device)
        # model = nn.DataParallel(model,device_ids=[0,1,2,3])
    """    
    if args.model == 'SwinLSTM-U':
        from SwinLSTM_U import SwinLSTM
        model = SwinLSTM(img_size=args.img_width, patch_size=args.patch_size,
                         in_chans=args.input_channels, embed_dim=args.embed_dim,
                         depths_downsample=args.depths_down, depths_upsample=args.depths_up,
                         num_heads=args.heads_number, window_size=args.window_size)
        model.to(args.device)
        model = nn.DataParallel(model,device_ids=[0,1,2,3,4,5,6,7])
    """  
    if args.model == 'SwinLSTM-M':
        from SwinLSTM_MU_new import SwinLSTM
        model = SwinLSTM(img_size=args.img_width, patch_size=args.patch_size,
                         in_chans=args.input_channels, embed_dim=args.embed_dim,
                         depths_downsample=args.depths_down, depths_upsample=args.depths_up,
                         num_heads=args.heads_number, window_size=args.window_size)
        # state_dict = torch.load('results/model/New_SwinLSTM_Unet_M_patch4_L8_C160_141_best')
        # parallel = False
        # if parallel:
        #     from collections import OrderedDict
        #     new_state_dict = OrderedDict()
        #     for k, v in state_dict.items():
        #         name = k[7:]  # remove `module.`
        #         new_state_dict[name] = v
        #     model.load_state_dict(new_state_dict)
        # else:
        #     model.load_state_dict(state_dict)
        model.to(args.device)
        # model = nn.DataParallel(model,device_ids=[0,1,2,3])
        
        
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=150, gamma=0.1)
    criterion = nn.MSELoss()
    layers =[2,10,16]
    # perceptual_loss = PerceptualLoss(layers).to(args.device)
    perceptual_loss = 0
    mseloss = nn.MSELoss()
    

    return model, perceptual_loss, mseloss, optimizer, scheduler



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
    model, perceptual_loss, mseloss, optimizer, scheduler = setup(args)
    cache_dir, model_dir, log_dir = make_dir(args)
    logger = init_logger(log_dir)
    best_val_loss = float('inf')
    for epoch in range(args.epochs):
        start_time = time.time()
        train_losses = []
        print(f'\n--- Epoch {epoch+1}/{args.epochs} ---')
        print('-----------------------------begin train-----------------------------')
        train_loss = train(args, logger, epoch, model, train_input_handle, perceptual_loss, mseloss, optimizer)
        
        plot_loss(train_losses, 'train', epoch, args.res_dir, 1)
        current_lr = optimizer.param_groups[0]['lr']
        if epoch%1 == 0:
            torch.save(model.state_dict(), f'{model_dir}/SwinLSTM_L6_{epoch}_MU')
        print(f'Time: {time.time() - start_time:.0f}s ; trainloss: {train_loss} ; lr: {current_lr}')
        logger.info(f'EP:{(int(epoch)):04d} Loss:{train_loss:.6f}')

        # scheduler.step()
        
        # 验证模型valid
        interval = 5
        
        start_time1 = time.time()
        valid_loss= test(args, logger, 0, model, valid_input_handle, mseloss, cache_dir)
        print(f'Time: {time.time() - start_time1:.0f}s ; validloss: {valid_loss}')
        
        if valid_loss < best_val_loss:
            best_val_loss = valid_loss
            best_model_state = model.state_dict()
            best_epoch = epoch
        if epoch % interval == 0 and epoch != 0:
            torch.save(best_model_state, f'{model_dir}/Swin_D_{int(best_epoch)}_best')
            print(f'[Best_Model every {interval} epoch]  Loss:{best_val_loss:.4f}')
            best_val_loss = float('inf')
            best_model_state = None

if __name__ == '__main__':

    print('Initializing models')
    
    main()

# def main():
#     # test
#     print('-----------------------------begin test-----------------------------')
#     set_seed(args.seed)
#     cache_dir, model_dir, log_dir = make_dir(args)
#     logger = init_logger(log_dir)

#     from SwinLSTM_D import SwinLSTM

#     model = SwinLSTM(img_size=args.img_width, patch_size=args.patch_size,
#                          in_chans=args.input_channels, embed_dim=args.embed_dim,
#                          depths_downsample=args.depths_down, depths_upsample=args.depths_up,
#                          num_heads=args.heads_number, window_size=args.window_size).to(args.device)

#     test_input_handle = Data_Center.data_provider(args.dataset_name, args.train_data_path,
#                                                   args.train_data_path, args.test_batch_size,
#                                                   args.img_width,
#                                                   seq_length=args.total_length, mode='test')
#     criterion = nn.MSELoss()

#     state_dict = torch.load('results/model/Swin_D_41_best')  # 读取 xxx.pth 模型
#     parallel = True
#     if parallel:
#         from collections import OrderedDict
#         new_state_dict = OrderedDict()
#         for k, v in state_dict.items():
#             name = k[7:]  # remove `module.`
#             new_state_dict[name] = v
#         model.load_state_dict(new_state_dict)
#     else:
#         model.load_state_dict(state_dict)

#     start_time = time.time()

#     _, mse, ssim = test(args, logger, 0, model, test_input_handle, criterion, cache_dir)

#     print(f'[Metrics]  MSE:{mse:.4f} SSIM:{ssim:.4f}')
#     print(f'Time usage per epoch: {time.time() - start_time:.0f}s')

# if __name__ == '__main__':
    
#     main()