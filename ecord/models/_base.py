from abc import ABC

import torch

class _DevicedModule(ABC):

    def set_devices(self, device, out_device):
        if device is None:
            if torch.cuda.is_available():
                self.device = 'cuda'
            else:
                self.device = 'cpu'
        else:
            # I trust you know what you are doing!
            self.device = device
        if out_device is None:
            out_device = self.device
        self.out_device = out_device
        self.to(self.device)