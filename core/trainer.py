import os.path
import datetime
import cv2
import numpy as np
# from skimage.measure import compare_ssim
from skimage.metrics import structural_similarity as compare_ssim
from core import utils
# import lpips
import torch
from PIL import Image

# loss_fn_alex = lpips.LPIPS(net='alex')

def train_one_batch(model,ims,real_input_flag,optimizer):
    loss = 0.0
    optimizer.zero_grad()
    pred_frames, loss_= model.train(ims, real_input_flag)
    optimizer.step()
    loss += loss_
    return loss

def valid(model,ims,gen_ims,configs,real_input_flag):
    valid_ims = gen_ims
    # img_gen获取pred值
    predims,batch_val_loss = model.valid(valid_ims, real_input_flag)
    predims_reshape_back = utils.reshape_patch_back(predims, configs.patch_size)
    output_length = configs.total_length - configs.input_length
    img_out = predims_reshape_back[:, -output_length:]
    # print(predims_reshape_back.shape,img_out.shape)#img_out.shape((batchsize, 2, 256, 256, 1))
    #   计算psnr;my_mae;
    batch_psnr = []
    my_mae = []

    new_ims = ims[:,:,:,:,:configs.img_channel]   #本意是想分出每个通道，但是本就是1

    for i in range(configs.total_length - configs.input_length):
        batch_psnr.append(0)
        my_mae.append(0)


    # batch_psnr
    for i in range(output_length):
        x = new_ims[:,i+configs.input_length,:,:,:]#真实值形式[bs,第几张图,H,W,1]
        gx = img_out[:,i,:,:,:]
        gx = np.maximum(gx, 0)
        gx = np.minimum(gx, 1)
        real_frm = np.uint8(x * 255)
        pred_frm = np.uint8(gx * 255)
        batch_psnr[i] += utils.batch_psnr(pred_frm, real_frm)

    # batch_motion_Criterion
    for batch in range(configs.batch_size):
        mae_ = []
        for i in range(output_length):
            # print(new_ims.shape)
            img_gt = np.uint8(new_ims[batch,i+configs.input_length,:,:,:]*255)
            # print(img_gt.shape)
            img_gt = np.reshape(img_gt,img_gt.shape[:-1])

            img_pd = img_out[batch, i, :, :, :]
            img_pd = np.maximum(img_pd, 0)
            img_pd = np.minimum(img_pd, 1)
            img_pd = np.uint8(img_pd * 255)
            # new_array = np.reshape(array, array.shape[:-1])
            img_pd = np.reshape(img_pd,img_pd.shape[:-1])
            img_gt_img = Image.fromarray(img_gt)
            img_pd_img = Image.fromarray(img_pd)
            #将每两张图片的mae计算添加到[]当中，同一序列的两张图进行一个平均，8(batch_size)个两数量的mae平均加入到[]当中
            # mae_.append(utils.batch_motion_Criterion(img_pd_img,img_gt_img))
        # my_mae.append(sum(mae_)/len(mae_))



    return batch_val_loss,batch_psnr,my_mae


def test(model,ims,gen_ims,configs,real_input_flag,batch_id):
    real_patch_ims = gen_ims
    pred_ims_patch,batch_test_loss = model.test(real_patch_ims,real_input_flag)
    pred_ims = utils.reshape_patch_back(pred_ims_patch,configs.patch_size)
    output_length = configs.total_length - configs.input_length
    pred_ims_out = pred_ims[:,-output_length:]

    #计算评价指标
    batch_psnr = []
    my_mae = []


    real_ims = ims[:, :, :, :, :configs.img_channel]

    for i in range(configs.total_length - configs.input_length):
        batch_psnr.append(0)
        my_mae.append(0)

    # batch_psnr
    for i in range(output_length):
        x = real_ims[:,i+configs.input_length,:,:,:]#真实值第i张图
        gx = pred_ims_out[:, i, :, :, :]
        gx = np.maximum(gx, 0)
        gx = np.minimum(gx, 1)
        real_frm = np.uint8(x * 255)
        pred_frm = np.uint8(gx * 255)
        batch_psnr[i] += utils.batch_psnr(pred_frm, real_frm)

    # batch_motion_Criterion
    # for batch in range(configs.batch_size):
    #     mae_ = []
    #     for i in range(output_length):
    #         # print(new_ims.shape)
    #         img_gt = np.uint8(real_ims[batch,i+configs.input_length,:,:,:]*255)
    #         # print(img_gt.shape)
    #         img_gt = np.reshape(img_gt,img_gt.shape[:-1])
    #
    #         img_pd = pred_ims_out[batch, i, :, :, :]
    #         img_pd = np.maximum(img_pd, 0)
    #         img_pd = np.minimum(img_pd, 1)
    #         img_pd = np.uint8(img_pd * 255)
    #         # new_array = np.reshape(array, array.shape[:-1])
    #         img_pd = np.reshape(img_pd,img_pd.shape[:-1])
    #         img_gt_img = Image.fromarray(img_gt)
    #         img_pd_img = Image.fromarray(img_pd)
    #         #将每两张图片的mae计算添加到[]当中，同一序列的两张图进行一个平均，8(batch_size)个两数量的mae平均加入到[]当中
    #         mae_.append(utils.batch_motion_Criterion(img_pd_img,img_gt_img))
    #     my_mae.append(sum(mae_)/len(mae_))

    #save preds&trues
    for img_id in range(configs.batch_size):
        save_path = os.path.join(configs.gen_frm_dir,str(batch_id),str(img_id))
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        for i in range(configs.total_length):
            name = 'trues' + str(i + 1) + '.png'
            file_name = os.path.join(save_path, name)
            img_gt = np.uint8(real_ims[img_id, i, :, :, :] * 255)
            cv2.imwrite(file_name, img_gt)
        for i in range(output_length):
            name = 'preds' + str(i + 1 + configs.input_length) + '.png'
            file_name = os.path.join(save_path, name)
            img_pd = pred_ims_out[img_id, i, :, :, :]
            img_pd = np.maximum(img_pd, 0)
            img_pd = np.minimum(img_pd, 1)
            img_pd = np.uint8(img_pd * 255)
            cv2.imwrite(file_name, img_pd)



    return batch_test_loss, batch_psnr, my_mae

