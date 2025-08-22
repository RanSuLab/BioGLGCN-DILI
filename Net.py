# -*- coding: utf-8 -*-
import torch
import math
import numpy as np
import torch.nn as nn
from layer import SparseGraphLearn, GraphConvolution, GATConvolution, GraphSAGE
from utils.utils import sample_mask, preprocess_Finaladj, preprocess_features

class BioGLGCN(nn.Module):
    def __init__(self, edge, adj, gene_p, num, input_dim, output_dim, phi, layer0_outputdim, layer1_outputdim, layer2_outputdim, dropout):
        super(BioGLGCN, self).__init__()

        self.x = 0
        self.S = 0
        self.outputs = 0
        self.edge = edge
        self.adj = adj
        self.gene_p = gene_p
        self.num = num
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.phi = phi
        self.layer0_outputdim = layer0_outputdim
        self.layer1_outputdim = layer1_outputdim
        self.layer2_outputdim = layer2_outputdim
        self.dropout = dropout

        self.relu = torch.nn.LeakyReLU()

        self.BatchNorm1d_1 = torch.nn.BatchNorm1d(num_features=978)

        self.lin1 = torch.nn.Linear(1024 * self.layer2_outputdim, 512)
        self.BatchNorm1d_2 = torch.nn.BatchNorm1d(num_features=512)
        
        self.lin2 = torch.nn.Linear(256, 64)
        self.BatchNorm1d_3 = torch.nn.BatchNorm1d(num_features=32)
        
        self.lin3 = torch.nn.Linear(64, 2)
        
    def forward(self, gl_input, gcn_input):

        self.x, self.S = self.layers0(gl_input)
        nn = gcn_input.float()

        x1 = self.layers1(nn, self.S)
        self.layer1 = self.BatchNorm1d_1(x1)
        
        self.layer2 = self.layers2(self.layer1, self.S)

        self.flattened = self.layer2.view(self.layer2.size(0), -1)

        self.dense1 = self.lin1(self.flattened)
        
        self.dense2 = self.lin2(self.dense1)
        
        self.dense3 = self.lin3(self.dense2)

        self.outputs = torch.nn.functional.softmax(self.dense3, dim=1)

        return self.outputs