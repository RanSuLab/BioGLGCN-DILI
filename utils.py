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


def preprocess_Finaladj(adj):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    adj_normalized =sp.coo_matrix(adj).tocoo()
    #The reference relationship between samples is expressed as the relationship between sample indexes
    edge = np.array(np.nonzero(adj_normalized.todense()))
    return sparse_to_tuple(adj_normalized), edge