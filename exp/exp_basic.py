import os
import torch
from models import Dynamic_TMoE

class Exp_Basic(object):
    def __init__(self, args):
        self.args = args
        self.model_dict = {
            'Dynamic_TMoE': Dynamic_TMoE
        }

        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device)

    def _build_model(self):
        raise NotImplementedError
        return None

    def _acquire_device(self):
        if hasattr(self.args, 'device'):
            device = self.args.device
            print('Use device: {}'.format(device))
            return device

        if self.args.use_gpu and self.args.gpu_type == 'cuda':
            if torch.cuda.is_available():
                os.environ["CUDA_VISIBLE_DEVICES"] = str(
                    self.args.gpu) if not self.args.use_multi_gpu else self.args.devices
                device = torch.device('cuda:{}'.format(self.args.gpu))
                print('Use GPU: cuda:{}'.format(self.args.gpu))
            else:
                device = torch.device('cpu')
                print('CUDA unavailable, use CPU')
        elif self.args.use_gpu and self.args.gpu_type == 'mps':
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = torch.device('mps')
                print('Use GPU: mps')
            else:
                device = torch.device('cpu')
                print('MPS unavailable, use CPU')
        else:
            device = torch.device('cpu')
            print('Use CPU')
        return device

    def _get_data(self):
        pass

    def vali(self):
        pass

    def train(self):
        pass

    def test(self):
        pass
