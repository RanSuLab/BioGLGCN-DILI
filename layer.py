# -*- coding: utf-8 -*-
import torch
import math
import numpy as np
import torch.nn as nn

from torch_geometric.nn import TopKPooling
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
import torch.nn.functional as F

from torch_geometric.nn import GCNConv, SAGEConv, GATConv

from torch.nn import Sequential as Seq, Linear, ReLU
from torch_geometric.nn import MessagePassing
from torch_geometric.nn import global_mean_pool
from torch_geometric.utils import remove_self_loops, add_self_loops, to_dense_adj, degree

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

torch.manual_seed(11)

def sparse_dropout(x, keep_prob):
    """Dropout for sparse tensors."""

    x_to_SparseTensor = torch.sparse_coo_tensor(indices=torch.tensor(x[0], dtype=torch.long).t(), values=x[1],
                                                size=x[2])
    x_to_SparseTensor = x_to_SparseTensor.to(device)
    # pre_out = torch.masked_select(x_to_SparseTensor, dropout_mask)

    return x_to_SparseTensor


def dot(x, y, sparse=False):
    """Wrapper for tf.matmul (sparse vs dense)."""
    if sparse:
        res = torch.sparse.mm(x, y)
    else:
        res = torch.matmul(x, y)
    return res


class GraphConvolution(nn.Module):
    '''Implement graph classification'''

    def __init__(self, input_dim, output_dim,
                 sparse_inputs=False, act=torch.nn.ReLU(), bias=False):
        super(GraphConvolution, self).__init__()

        self.vars = nn.ParameterDict()

        self.act = act
        self.sparse_inputs = sparse_inputs
        self.bias = bias
        self.output_dim = output_dim

        self.weights_initializer = nn.init.xavier_uniform_
        self.biases_initializer = nn.init.zeros_

        self.vars['weights'] = nn.Parameter(self.weights_initializer(torch.empty(input_dim, output_dim), gain=math.sqrt(2.0)))
        self.vars['aggregateInfo_weights'] = nn.Parameter(self.weights_initializer(torch.empty(output_dim*2, output_dim), gain=math.sqrt(2.0)))
        if self.bias:
            self.vars['bias'] = nn.Parameter(self.biases_initializer(torch.empty(output_dim)))

    def forward(self, inputs, adj):

        x = inputs
        adj = adj.to(torch.float)
        inputs_unstacked = torch.unbind(x)
        def fn(x_slice):
            pre_sup = dot(x_slice, self.vars['weights'], sparse=self.sparse_inputs) # sparse=False
            output = dot(adj, pre_sup, sparse=True)
            if self.bias:
                output = output + self.vars['bias']
            return output

        outputs = torch.stack([fn(i) for i in inputs_unstacked])
        return self.act(outputs)

class GATConvolution(nn.Module):
    '''Implement graph classification'''

    def __init__(self, input_dim, output_dim,
                 sparse_inputs=False, act=torch.nn.ReLU(), bias=False):
        super(GATConvolution, self).__init__()

        self.vars = nn.ParameterDict()

        self.act = act
        self.sparse_inputs = sparse_inputs
        self.bias = bias
        self.output_dim = output_dim

        self.weights_initializer = nn.init.xavier_uniform_
        self.a_initializer = nn.init.xavier_uniform_
        self.biases_initializer = nn.init.zeros_

        # self.weights = nn.Parameter(weights_initializer(torch.empty(input_dim, output_dim)))

        self.vars['weights'] = nn.Parameter(self.weights_initializer(torch.empty(input_dim, output_dim), gain=math.sqrt(2.0)))
        self.vars['a'] = nn.Parameter(self.a_initializer(torch.FloatTensor(output_dim * 2, 1)))  # ćéĺé a

        if self.bias:
            self.vars['bias'] = nn.Parameter(self.biases_initializer(torch.empty(output_dim)))

    def forward(self, inputs, adj):

        x = inputs

        adj = adj.to(torch.float)
        inputs_unstacked = torch.unbind(x)
        def fn(x_slice):
            wh = dot(x_slice, self.vars['weights'], sparse=self.sparse_inputs)
            e = torch.mm(wh, self.vars['a'][: self.output_dim]) + torch.matmul(wh, self.vars['a'][self.output_dim:]).T
            e = torch.nn.LeakyReLU()(e)
            attention = torch.where(adj.to_dense() > 0, e, -1e9 * torch.ones_like(e))
            attention = F.softmax(attention, dim=1)

            output = torch.mm(attention, wh)
            output = dot(adj, output, sparse=True)

            if self.bias:
                output = output + self.vars['bias']
            return output

        outputs = torch.stack([fn(i) for i in inputs_unstacked])
        
        return self.act(outputs)
    


class GraphSAGE(nn.Module):

    def __init__(self, input_dim, output_dim,
                 sparse_inputs=False, act=torch.nn.ReLU(), bias=False):
        super(GraphSAGE, self).__init__()

        self.act = act
        self.sparse_inputs = sparse_inputs
        self.bias = bias
        self.output_dim = output_dim

        self.sage1 = SAGEConv(input_dim, output_dim)

    def forward(self, inputs, adj):

        x = inputs
        adj = adj.to(torch.float)

        inputs_unstacked = torch.unbind(x)

        def fn(x_slice):
            output = self.sage1(x_slice, adj.indices())

            return output

        outputs = torch.stack([fn(i) for i in inputs_unstacked])
        
        return self.act(outputs)
