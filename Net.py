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

        self.layers0 = SparseGraphLearn(input_dim=self.input_dim,
                                        output_dim=self.layer0_outputdim,
                                        num=self.num,
                                        edge=self.edge,
                                        adj=self.adj,
                                        gene_p=self.gene_p,
                                        phi=self.phi,
                                        act=torch.nn.Sigmoid(),
                                        sparse_inputs=True)

        self.layers1 = GraphConvolution(input_dim=1,
                                        output_dim=self.layer1_outputdim,
                                        act=torch.nn.LeakyReLU(),
                                        sparse_inputs=False,
                                        bias=True)

        self.layers2 = GraphConvolution(input_dim=self.layer1_outputdim,
                                        output_dim=self.layer2_outputdim,
                                        act=torch.nn.LeakyReLU(),
                                        sparse_inputs=False,
                                        bias=True
                                        )

        self.BatchNorm1d_1 = torch.nn.BatchNorm1d(num_features=978)

        self.lin1 = torch.nn.Linear(978 * self.layer2_outputdim, 512)
        self.BatchNorm1d_2 = torch.nn.BatchNorm1d(num_features=512)
        
        self.lin2 = torch.nn.Linear(512, 32)
        self.BatchNorm1d_3 = torch.nn.BatchNorm1d(num_features=32)
        
        self.lin3 = torch.nn.Linear(32, 2)
        
    def forward(self, gl_input, gcn_input):

        self.x, self.S = self.layers0(gl_input)
        nn = gcn_input.float()

        x1 = self.layers1(nn, self.S)
        self.layer1 = self.BatchNorm1d_1(x1)
        
        self.layer2 = self.layers2(self.layer1, self.S)

        self.flattened = self.layer2.view(self.layer2.size(0), -1)

        self.dense1 = self.lin1(self.flattened)
        self.dense1 = self.BatchNorm1d_2(self.dense1)
        self.dense1 = self.relu(self.dense1)
        self.dense1 = torch.nn.Dropout(p=self.dropout)(self.dense1)

        self.dense2 = self.lin2(self.dense1)
        self.dense2 = self.BatchNorm1d_3(self.dense2)
        self.dense2 = self.relu(self.dense2)
        self.dense2 = torch.nn.Dropout(p=self.dropout)(self.dense2)
        
        self.dense3 = self.lin3(self.dense2)

        self.outputs = torch.nn.functional.softmax(self.dense3, dim=1)

        return self.outputs