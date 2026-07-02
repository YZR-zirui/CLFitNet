import numpy as np
import torch
from torch.cuda import amp
from torch.cuda.amp import autocast as autocast
from utils import compute_metrics, visualize
from tqdm import tqdm
from dataAug import apply_augmentations

scaler = torch.amp.GradScaler('cuda')

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
    # num_layers表示有几层的意思
    states_down = [None] * len(num_layers)
    states_up = [None] * len(num_layers)

    outputs = []

    inputs_len = inputs.shape[1]

    last_input = inputs[:, -1]

    for i in range(inputs_len - 1):     #i代表是inputs_len，体现第几张图，横向结构
        output, states_down, states_up = model(inputs[:, i], states_down, states_up)    #states_down是一个列表，第一个表示H，第二个表示C；这里结构和PredRNN很像
        outputs.append(output)

    for i in range(targets_len):    #预测后十张
        output, states_down, states_up = model(last_input, states_down, states_up)
        outputs.append(output)
        last_input = output

    return outputs

def model_forward_multi_layer_U(model, inputs, targets_len, num_layers):
    # num_layers表示有几层的意思
    states_down = [None] * len(num_layers)
    states_up = [None] * len(num_layers)

    outputs = []

    inputs_len = inputs.shape[1]

    last_input = inputs[:, -1]

    for i in range(inputs_len - 1):     #i代表是inputs_len，体现第几张图，横向结构
        output, states_down, states_up = model(inputs[:, i], states_down, states_up)    #states_down是一个列表，第一个表示H，第二个表示C；这里结构和PredRNN很像
        outputs.append(output)

    for i in range(targets_len):    #预测后十张
        output, states_down, states_up = model(last_input, states_down, states_up)
        outputs.append(output)
        last_input = output

    return outputs

def train(args, logger, epoch, model, train_input_handle, perceptual_loss, mseloss, optimizer):
    
    model.train()
    num_batches = train_input_handle.total_batch()
    losses = []
    torch.autograd.set_detect_anomaly(True)
    pbar = tqdm(range(num_batches))
    for batch_idx in pbar:
        optimizer.zero_grad()
        if train_input_handle.no_batch_left():
            train_input_handle.begin(do_shuffle=True)
        ims = train_input_handle.get_batch()
        ims = apply_augmentations(ims)
        # print(ims.shape)
        inputs = ims[:, 0:3, :, :, :]   # 输入0，1，2
        targets = ims[:, 3:5, :, :, :]  # 输出对比的真是值3，4
        inputs = torch.FloatTensor(inputs).to(args.device)
        targets = torch.FloatTensor(targets).to(args.device)

        inputs = inputs.permute(0, 1, 4, 2, 3).contiguous()
        targets = targets.permute(0, 1, 4, 2, 3).contiguous()
        targets_len = targets.shape[1]
        
        with torch.amp.autocast('cuda'):

            outputs = model_forward_multi_layer_U(model, inputs, targets_len, args.depths_down)
            outputs = torch.stack(outputs).permute(1, 0, 2, 3, 4).contiguous()
            targets_ = torch.cat((inputs[:, 1:], targets), dim=1)

            if batch_idx == 0:
                print(f"Outputs size: {outputs.size()}")
                print(f"Targets size: {targets_.size()}")

            loss = mseloss(outputs, targets_)
        # loss.backward()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        # optimizer.step()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        losses.append(loss.item())

        train_input_handle.next()

    pbar.close()
    return np.mean(losses)



def test(args, logger, epoch, model, test_input_handle, criterion, cache_dir):
    model.eval()
    num_batches = test_input_handle.total_batch()
    losses, mses, ssims = [], [], []
    for batch_idx in tqdm(range(num_batches)):
        if test_input_handle.no_batch_left():
            test_input_handle.begin(do_shuffle=False)
        ims = test_input_handle.get_batch()
        inputs = ims[:, 0:3, :, :, :]  # 输入0，1，2
        targets = ims[:, 3:5, :, :, :]  # 输出对比的真是值3，4
        inputs = torch.FloatTensor(inputs).to(args.device)
        targets = torch.FloatTensor(targets).to(args.device)

        inputs = inputs.permute(0, 1, 4, 2, 3).contiguous()
        targets = targets.permute(0, 1, 4, 2, 3).contiguous()

        with torch.no_grad():
            # inputs, targets = map(lambda x: x.float().to(args.device), [inputs, targets])
            targets_len = targets.shape[1]
            
            outputs = model_forward_multi_layer_U(model, inputs, targets_len, args.depths_down)

            outputs = torch.stack(outputs).permute(1, 0, 2, 3, 4).contiguous()
            targets_ = torch.cat((inputs[:, 1:], targets), dim=1)

            losses.append(criterion(outputs, targets_).item())

            inputs_len = inputs.shape[1]
            outputs = outputs[:, inputs_len - 1:]

            # mse, ssim, psnr = compute_metrics(outputs, targets)

            # mses.append(mse)
            # ssims.append(ssim)
            # psnrs.append(psnr)

            if batch_idx and batch_idx % args.log_test == 0:
                # logger.info(f'EP:{epoch:04d} BI:{batch_idx:03d}/{num_batches:03d} Loss:{np.mean(losses):.6f} MSE:{mse:.4f} SSIM:{ssim:.4f} PSNR:{psnr:.4f}')
                visualize(inputs, targets, outputs, epoch, batch_idx, cache_dir)
                # gen_image(targets,outputs,epoch,batch_idx)
        test_input_handle.next()

    return np.mean(losses)
