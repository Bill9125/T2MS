import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR
from torch.optim.lr_scheduler import SequentialLR
from abc import ABC, abstractmethod


class BaseModel(nn.Module, ABC):
    def __init__(self):
        super().__init__()
    @abstractmethod
    def shared_eval(self, batch, optimizer, scheduler, mode):
        pass

    def configure_optimizers(self, lr=1e-3):
        optimizer = torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=1e-2)
        scheduler1 = LinearLR(optimizer, start_factor=0.1, total_iters=1000)  # warmup
        scheduler2 = CosineAnnealingLR(optimizer, T_max=400-1000, eta_min=1e-6)
        scheduler = SequentialLR(optimizer, schedulers=[scheduler1, scheduler2], milestones=[1000])
        return optimizer, scheduler
