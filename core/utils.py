import numpy as np
from PIL import Image

def reshape_patch(img_tensor, patch_size):
    assert 5 == img_tensor.ndim,'维度不为5'#断言，img_tensor的维度，若维度为5则不报错
    batch_size = np.shape(img_tensor)[0]#B
    seq_length = np.shape(img_tensor)[1]#N
    img_height = np.shape(img_tensor)[2]#H
    img_width = np.shape(img_tensor)[3]#W
    num_channels = np.shape(img_tensor)[4]#C
    a = np.reshape(img_tensor, [batch_size, seq_length,
                                img_height//patch_size, patch_size,
                                img_width//patch_size, patch_size,
                                num_channels])
    b = np.transpose(a, [0,1,2,4,3,5,6])
    patch_tensor = np.reshape(b, [batch_size, seq_length,
                                  img_height//patch_size,
                                  img_width//patch_size,
                                  patch_size*patch_size*num_channels])#??????
    return patch_tensor

def reshape_patch_back(patch_tensor, patch_size):
    assert 5 == patch_tensor.ndim
    batch_size = np.shape(patch_tensor)[0]
    seq_length = np.shape(patch_tensor)[1]
    patch_height = np.shape(patch_tensor)[2]
    patch_width = np.shape(patch_tensor)[3]
    channels = np.shape(patch_tensor)[4]
    img_channels = channels // (patch_size*patch_size)
    a = np.reshape(patch_tensor, [batch_size, seq_length,
                                  patch_height, patch_width,
                                  patch_size, patch_size,
                                  img_channels])
    b = np.transpose(a, [0,1,2,4,3,5,6])
    img_tensor = np.reshape(b, [batch_size, seq_length,
                                patch_height * patch_size,
                                patch_width * patch_size,
                                img_channels])
    return img_tensor

def batch_psnr(pred_frm, real_frm):
    if pred_frm.ndim == 3:# ndim:number of dimensions数组维数
        axis = (1, 2)
    elif pred_frm.ndim == 4:
        axis = (1, 2, 3)# hwc三个维度进行计算
    x = np.int32(pred_frm)
    y = np.int32(real_frm)
    num_pixels = float(np.size(pred_frm[0]))
    mse = np.sum((x - y) ** 2, axis=axis, dtype=np.float32) / num_pixels
    psnr = 20 * np.log10(255) - 10 * np.log10(mse)
    return np.mean(psnr)



def batch_motion_Criterion(pred_frm, real_frm):
    # pred_frm.show()
    # real_frm.show()
    # print(type(pred_frm),type(real_frm))
    array_real = np.array(real_frm)
    array_pred = np.array(pred_frm)
    y, x = array_real.shape

    #list_img记录位置上的像素值
    list_img1 = []
    list_img2 = []
    list_img2_i = []


    for i in range(x):
        if i == 0:
            list_img2_i.append(i)
            for j in range(y - 1):
                if array_real[j, i] == 51 and array_real[j - 1, i] == 0:
                    list_img1.append(array_real[j, i])
                    list_img2.append(array_pred[j, i])

        if i == 32:
            list_img2_i.append(i)
            for j in range(y - 1):
                if array_real[j, i] == 51 and array_real[j - 1, i] == 0:
                    list_img1.append(array_real[j, i])
                    list_img2.append(array_pred[j, i])

        if i == 64:
            list_img2_i.append(i)
            for j in range(y - 1):
                if array_real[j, i] == 51 and array_real[j - 1, i] == 0:
                    list_img1.append(array_real[j, i])
                    list_img2.append(array_pred[j, i])

        if i == 96:
            list_img2_i.append(i)
            for j in range(y - 1):
                if array_real[j, i] == 51 and array_real[j - 1, i] == 0:
                    list_img1.append(array_real[j, i])
                    list_img2.append(array_pred[j, i])
        if i == 128:
            list_img2_i.append(i)
            for j in range(y - 1):
                if array_real[j, i] == 51 and array_real[j - 1, i] == 0:
                    list_img1.append(array_real[j, i])
                    list_img2.append(array_pred[j, i])
        if i == 160:
            list_img2_i.append(i)
            for j in range(y - 1):
                if array_real[j, i] == 51 and array_real[j - 1, i] == 0:
                    list_img1.append(array_real[j, i])
                    list_img2.append(array_pred[j, i])
        if i == 192:
            list_img2_i.append(i)
            for j in range(y - 1):
                if array_real[j, i] == 51 and array_real[j - 1, i] == 0:
                    list_img1.append(array_real[j, i])
                    list_img2.append(array_pred[j, i])
        if i == 224:
            list_img2_i.append(i)
            for j in range(y - 1):
                if array_real[j, i] == 51 and array_real[j - 1, i] == 0:
                    list_img1.append(array_real[j, i])
                    list_img2.append(array_pred[j, i])
        if i == 255:
            list_img2_i.append(i)
            for j in range(y - 1):
                if array_real[j, i] == 51 and array_real[j - 1, i] == 0:
                    list_img1.append(array_real[j, i])
                    list_img2.append(array_pred[j, i])
    # print(list_img2, list_img1)
    mae = 0
    num = 0
    for y_pred, y_true in zip(list_img1, list_img2):
        # print(y_pred, y_true)
        d = np.abs(y_pred - y_true)
        mae += d.tolist()
        num += 1
    # print(mae)
    # 计算MAE
    MAE = mae / num
    # 输出结果
    # print(f'The MAE between the two images is: {MAE}')
    return MAE

