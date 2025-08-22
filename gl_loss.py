# -*- coding: utf-8 -*-

import torch
import math
import numpy as np
import torch.nn as nn
from torch_geometric.nn import TopKPooling
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
import torch.nn.functional as F
from torch.nn import Sequential as Seq, Linear, ReLU

torch.autograd.set_detect_anomaly(True)

# 自定义损失函数
class glLoss(nn.Module):
    def __init__(self, model, weight_decay, L1_gamma, L1_beta, phi=25):
        super(glLoss, self).__init__()

        self.model = model
        self.weight_decay = weight_decay
        self.L1_gamma = L1_gamma
        self.L1_beta = L1_beta
        self.phi = phi

    def forward(self):

        self.loss = torch.tensor(0, dtype=torch.float, requires_grad=True)
        self.loss1 = torch.tensor(0, dtype=torch.float, requires_grad=True)

        for var in self.model.layers0.vars.values():
            # var = var.detach()
            self.loss1 = self.loss1 + self.weight_decay * torch.norm(var) ** 2

        # Graph Learning loss
        device = self.model.S.device
        D = (torch.diag(torch.ones(self.model.num)) * 1).to(device)
        

        self.loss1 = self.loss1 + torch.trace(torch.matmul(D, x)) * 0.1

        self.loss1 = self.loss1 - torch.trace(torch.sparse.mm(torch.transpose(S, 0, 1), S.to_dense())) * self.L1_gamma


        N = self.model.num
        adj = self.model.adj


        self.loss = self.loss1

        return self.loss
