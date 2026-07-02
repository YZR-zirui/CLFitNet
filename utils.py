import os
import time
import torch
import random
import logging
import matplotlib
import numpy as np
from matplotlib import pyplot as plt
from skimage.metrics import structural_similarity, peak_signal_noise_ratio
import cv2
from torch.utils import checkpoint

matplotlib.use('agg')

def set_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True

def gen_image(targets, outputs,epoch,batch_idx):
    src = './gen_image'
    for t in range(outputs.shape[1]):
        result = src + '/' + str(epoch) + str(batch_idx) + '.png'
        output_out = outputs[0, t].detach().cpu().numpy()
        output_out = (output_out * 255).astype(np.uint8)
        output_out = output_out.transpose(1, 2, 0)
        cv2.imwrite(result, output_out)


def visualize(inputs, targets, outputs, epoch, idx, cache_dir):

    _, axarray = plt.subplots(3, inputs.shape[1], figsize=(inputs.shape[1] * 5, 10))
    for t in range(inputs.shape[1]):
        axarray[0][t].imshow(inputs[0, t, 0].detach().cpu().numpy(), cmap='gray')
    for t in range(targets.shape[1]):
        axarray[1][t].imshow(targets[0, t, 0].detach().cpu().numpy(), cmap='gray')
        axarray[2][t].imshow(outputs[0, t, 0].detach().cpu().numpy(), cmap='gray')

    plt.savefig(os.path.join(cache_dir, '{:03d}-{:03d}.png'.format(epoch, idx)))
    plt.close()

def plot_loss(loss_records, loss_type, epoch, plot_dir, step):
    plt.plot(range((epoch + 1) // step), loss_records, label=loss_type)
    plt.legend()
    plt.savefig(os.path.join(plot_dir, '{}_loss_records.png'.format(loss_type)))
    plt.close()

def MAE(pred, true):
    return np.mean(np.abs(pred - true), axis=(0, 1)).sum()
    
def MSE(pred, true):
    return np.mean((pred - true) ** 2, axis=(0, 1)).sum()

# cite the 'PSNR' code from E3D-LSTM, Thanks!
# https://github.com/google/e3d_lstm/blob/master/src/trainer.py line 39-40
def PSNR(pred, true):
    mse = np.mean((np.uint8(pred * 255) - np.uint8(true * 255)) ** 2)
    return 20 * np.log10(255) - 10 * np.log10(mse)

def compute_metrics(predictions, targets):
    targets = targets.permute(0, 1, 3, 4, 2).detach().cpu().numpy()
    predictions = predictions.permute(0, 1, 3, 4, 2).detach().cpu().numpy()

    batch_size = predictions.shape[0]
    Seq_len = predictions.shape[1]

    ssim = 0
    psnr = 0

    src = './gen_images'
    for batch in range(batch_size):
        for frame in range(Seq_len):

            ssim += structural_similarity(targets[batch, frame].squeeze(), predictions[batch, frame].squeeze() , data_range=1.0)

            psnr += peak_signal_noise_ratio(targets[batch, frame].squeeze(), predictions[batch, frame].squeeze())
    ssim /= (batch_size * Seq_len)
    psnr /= (batch_size * Seq_len)
    mse = MSE(predictions, targets)

    return mse, ssim, psnr

def check_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def make_dir(args):

    cache_dir = os.path.join(args.res_dir, 'cache')
    check_dir(cache_dir)

    model_dir = os.path.join(args.res_dir, 'model')
    check_dir(model_dir)

    log_dir = os.path.join(args.res_dir, 'log')
    check_dir(log_dir)

    return cache_dir, model_dir, log_dir

def init_logger(log_dir):
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                        datefmt='%m-%d %H:%M',
                        filename=os.path.join(log_dir, time.strftime("%Y_%m_%d") + '.log'),
                        filemode='w')

    # console = logging.StreamHandler()
    console = logging.FileHandler(filename=os.path.join(log_dir, time.strftime("%Y_%m_%d") + '.log'))
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)

    return logging

if __name__ == '__main__':
    print(checkpoint.keys())
    # log_dir = os.path.join('./results', 'log')
    # logger = init_logger(log_dir)
    # logger.info(',,')
