import torch
import torch.nn as nn
from Gate_Unit.SpatioTemporalLSTMCell import SpatioTemporalLSTMCell


class RNN(nn.Module):
    def __init__(self, num_layers, num_hidden, configs):
        super(RNN, self).__init__()

        self.configs = configs
        self.frame_channel = configs.patch_size * configs.patch_size * configs.img_channel  #inchannel=framechannel=4*4*1
        self.num_layer = num_layers
        self.num_hidden = num_hidden
        cell_list = []

        width = configs.img_width // configs.patch_size
        self.MSE_criterion = nn.MSELoss()
        # 从这个for循环里可以看出，num_layers是CELL的个数
        for i in range(num_layers):
            if i == 0:
                in_channel = self.frame_channel
            else:
                in_channel = num_hidden[i - 1]
            cell_list.append(
                SpatioTemporalLSTMCell(in_channel, num_hidden[i], width, configs.filter_size, configs.stride, configs.layer_norm)
            )
        self.cell_list = nn.ModuleList(cell_list)
        self.conv_last = nn.Conv2d(in_channels=num_hidden[num_layers-1],out_channels=self.frame_channel,kernel_size=1,stride=1,padding=0,bias=False)

    def forward(self,frames_tensor,mask_true):
        # [batch, length, height, width, channel] -> [batch, length, channel, height, width]
        frames = frames_tensor.permute(0, 1, 4, 2, 3).contiguous()
        mask_true = mask_true.permute(0, 1, 4, 2, 3).contiguous()

        batch = frames.shape[0]
        height = frames.shape[3]
        width = frames.shape[4]

        next_frames = []
        h_t = []
        c_t = []

        for i in range(self.num_layer):
            zeros = torch.zeros([batch,self.num_hidden[i],height,width]).to(self.configs.device)
            h_t.append(zeros)
            c_t.append(zeros)

        memory = torch.zeros([batch,self.num_hidden[0],height,width]).to(self.configs.device)
        '''
        ·反向计划抽样 reverse_scheduled_sampling RSS,用于时序编码，在训练时通过随即隐藏真实观察结果的概率，强制模型了解更多关于长期动态的信息
        ·计划抽样 scheduled_sampling SS,用于预测编码，以减轻训练和推理阶段之间数据流的不一致性。
        训练时，ENcoder用RSS，DEcoder用SS
        '''
        for t in range(self.configs.total_length-1):
            if self.configs.reverse_scheduled_sampling == 1:
                if t == 0:
                    net = frames[:, t]
                else:
                    net = mask_true[:, t - 1] * frames[:, t] + (1 - mask_true[:, t - 1]) * x_gen
            else:
                if t < self.configs.input_length:
                    net = frames[:, t]
                else:
                    net = mask_true[:, t - self.configs.input_length] * frames[:, t] + \
                          (1 - mask_true[:, t - self.configs.input_length]) * x_gen
            h_t[0], c_t[0], memory = self.cell_list[0](net, h_t[0], c_t[0], memory)

            for i in range(1, self.num_layer):
                h_t[i], c_t[i], memory = self.cell_list[i](h_t[i - 1], h_t[i], c_t[i], memory)

            x_gen = self.conv_last(h_t[self.num_layer - 1])
            # 在此处添加边缘引导模块，X_gen是输出的图像
            next_frames.append(x_gen)

        # [length, batch, channel, height, width] -> [batch, length, height, width, channel]
        next_frames = torch.stack(next_frames, dim=0).permute(1, 0, 3, 4, 2).contiguous()
        loss = self.MSE_criterion(next_frames, frames_tensor[:, 1:])
        return next_frames, loss
