# -*- coding: utf-8 -*-

import torch
import math
import numpy as np
import torch.nn as nn

from torch_geometric.nn import TopKPooling
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
import torch.nn.functional as F

from torch.nn import Sequential as Seq, Linear, ReLU

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

torch.autograd.set_detect_anomaly(True)

# 自定义损失函数
class gcnLoss(nn.Module):
    def __init__(self, model, weight_decay):
        super(gcnLoss, self).__init__()

        self.model = model
        self.weight_decay = weight_decay

    def forward(self, outputs, y_train):

        self.loss = torch.tensor(0, dtype=torch.float, requires_grad=True).to(device)
        self.loss1 = torch.tensor(0, dtype=torch.float, requires_grad=True).to(device)
        self.loss2 = torch.tensor(0, dtype=torch.float, requires_grad=True).to(device)

        for var in self.model.layers1.vars.values():
            # var = var.detach()
            self.loss1 = self.loss1 + self.weight_decay * torch.norm(var) ** 2
        for var in self.model.layers2.vars.values():
            # var = var.detach()
            self.loss2 = self.loss2 + self.weight_decay * torch.norm(var) ** 2

        weights = [1.0, 1.0]
        class_weights = torch.FloatTensor(weights).to(device)
        criterion = torch.nn.CrossEntropyLoss(weight=class_weights)

        # criterion = torch.nn.CrossEntropyLoss()
        self.loss2 = self.loss2 + criterion(outputs, y_train.detach())
        self.loss = self.loss1 + self.loss2

        return self.loss