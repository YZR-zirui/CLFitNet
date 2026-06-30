import os
import torch
from torch.optim import Adam
from models import predrnn
from torch.optim.lr_scheduler import OneCycleLR
import datetime

class Model(object):
    def __init__(self, configs):
        self.configs = configs
        self.num_hidden = [int(x) for x in configs.num_hidden.split(',')]
        self.num_layers = len(self.num_hidden)
        networks_map = {
            'predrnn': predrnn.RNN,
        }

        if configs.model_name in networks_map:
            Network = networks_map[configs.model_name]
            self.network = Network(self.num_layers, self.num_hidden, configs).to(configs.device)
        else:
            raise ValueError('Name of network unknown %s' % configs.model_name)

    def save(self, epoch):
        print("saving model")
        stats = {}
        stats['net_param'] = self.network.state_dict()
        checkpoint_path = os.path.join(self.configs.save_dir, 'model512.ckpt'+'_'+str(epoch))
        torch.save(stats, checkpoint_path)
        print("save model to %s" % checkpoint_path)

    def load(self, checkpoint_path):
        print('load model:', checkpoint_path)
        stats = torch.load(checkpoint_path)
        self.network.load_state_dict(stats['net_param'])

    def train(self, frames, mask):
        frames_tensor = torch.FloatTensor(frames).to(self.configs.device)
        mask_tensor = torch.FloatTensor(mask).to(self.configs.device)

        # self.optimizer.zero_grad()
        pred_frames, loss = self.network(frames_tensor, mask_tensor)
        loss.backward()
        # self.optimizer.step()
        # self.scheduler.step()
        # return pred_frames,loss.detach().cpu().numpy(),self.scheduler.get_last_lr()[0]
        return pred_frames,loss.detach().cpu().numpy()

    def test(self, frames, mask):
        frames_tensor = torch.FloatTensor(frames).to(self.configs.device)
        mask_tensor = torch.FloatTensor(mask).to(self.configs.device)
        pred_frames, loss = self.network(frames_tensor, mask_tensor)
        return pred_frames.detach().cpu().numpy(),loss

    def valid(self,frames,mask):
        frames_tensor = torch.FloatTensor(frames).to(self.configs.device)
        mask_tensor = torch.FloatTensor(mask).to(self.configs.device)
        pred_frames, loss = self.network(frames_tensor, mask_tensor)
        return pred_frames.detach().cpu().numpy(),loss



