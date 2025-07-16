# -*- coding: utf-8 -*-

import numpy as np
import scipy.sparse as sp
import torch

torch.manual_seed(123)


def normalize_list(lst):
    max_val = max(lst)
    min_val = min(lst)
    normalized_lst = [(x - min_val) / (max_val - min_val) for x in lst]
    return normalized_lst


def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def confusionmetrics(y_pred,y_actual):
    from sklearn.metrics import confusion_matrix
    confusion_matrix = confusion_matrix(y_actual, y_pred)
    TP = confusion_matrix[0, 0]
    FN = confusion_matrix[0, 1]
    FP = confusion_matrix[1, 0]
    TN = confusion_matrix[1, 1]
    accuracy = (TP +TN) / (TP + FP + FN + TN)
    tpr = TP/ (TP + FN)
    fpr = FP / (FP + TN)
    fnr = FN/ (TP + FN)
    tnr = TN / (FP + TN)
    specificity=tnr
    recall=tpr
    sensitivity=tpr
    precision =TP / (TP  + FP)
    f1_score = (2 * (precision * recall)) / (precision + recall)
    return specificity,sensitivity, f1_score, accuracy

# mask索引为idx的位置被设置为1，其他位置为0
def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    # print(mask)
    return np.array(mask, dtype=np.bool_)

def sparse_to_tuple(sparse_mx):
    """Convert sparse matrix to tuple representation."""
    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        shape = mx.shape
        return coords, values, shape
    if isinstance(sparse_mx, list):
    # 如果 sparse_mx 是 list 类型，那么 isinstance(sparse_mx, list) 将返回 True，否则返回 False
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx

#normalization
def preprocess_features(features):
    features = sp.coo_matrix(features)
    
    rowsum = np.array(features.sum(1)) # Sum over each row
    r_inv = np.power(rowsum, -1).flatten() # 取倒数
    r_inv[np.isinf(r_inv)] = 0.# If all of the rows are 0, then r_inv is going to be equal to infinity, so let's set r_inv to 0 for those rows
    r_mat_inv = sp.diags(r_inv)# Construct the diagonal matrix with the diagonal element r_inv
    features = r_mat_inv.dot(features)  # By standardizing the dot product of the diagonal matrix with the original matrix, each row of the original matrix is multiplied by the corresponding r_inv, which is equivalent to dividing by sum
    return sparse_to_tuple(features)




def preprocess_Finaladj(adj):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    adj_normalized =sp.coo_matrix(adj).tocoo()
    #The reference relationship between samples is expressed as the relationship between sample indexes
    edge = np.array(np.nonzero(adj_normalized.todense()))
    return sparse_to_tuple(adj_normalized), edge