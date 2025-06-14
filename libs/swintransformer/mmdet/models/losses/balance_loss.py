# core/loss/balance_loss.py
import mmcv
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..builder import LOSSES
@LOSSES.register_module()
class BalanceLoss(nn.Module):
    def __init__(self, loss_weight=0.5):
        super().__init__()
        self.loss_weight = loss_weight

    def forward(self, balance_loss):
        return self.loss_weight * balance_loss