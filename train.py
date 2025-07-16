# -*- coding: utf-8 -*-
from __future__ import division
from __future__ import print_function

import warnings
warnings.filterwarnings("ignore")

import os
import time
import math
import pickle
import argparse
import torch
# torch.cuda.empty_cache()  # 清空缓存
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from sklearn.manifold import TSNE
from sklearn.metrics import auc, roc_curve, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, precision_recall_curve
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
KF = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=1024)
from sklearn.model_selection import ParameterGrid

from collections import defaultdict

from Net import BioGLGCN
from gl_loss import glLoss
from gcn_loss import gcnLoss
from utils.utils import normalize_list, sample_mask, preprocess_Finaladj, preprocess_features

import visdom
viz_name = 'BioGL-GCN_model'
viz = visdom.Visdom(env=viz_name, use_incoming_socket=False)

parser = argparse.ArgumentParser(description='Train BioGL-GCN with custom hyperparameters')
parser.add_argument('--gl_lr', type=float, default=0.0001, help='Layer0 learning rate')
parser.add_argument('--layer0_outputdim', type=int, default=90, help='Output dim of layer0')

parser.add_argument('--gcn_lr', type=float, default=0.00002, help='Layer1+Linear learning rate')
parser.add_argument('--layer1_outputdim', type=int, default=4, help='Output dim of layer1')
parser.add_argument('--layer2_outputdim', type=int, default=8, help='Output dim of layer2')

parser.add_argument('--dropout', type=float, default=0.3, help='Dropout rate')

parser.add_argument('--L1_gamma', type=float, default=1e-5, help='L1 regularization weight 1')
parser.add_argument('--L1_beta', type=float, default=1e-5, help='L1 regularization weight 2')
parser.add_argument('--weight_decay', type=float, default=1e-8, help='Weight Decay')

parser.add_argument('--gamma', type=float, default=0.9, help='L1 regularization weight 1')

parser.add_argument('--Phi', type=int, default=25, help='Power of matrix A')

parser.add_argument('--gpu', type=int, default=0, help='GPU id to use')
parser.add_argument('--epoch', type=int, default=100, help='train epoch')
args = parser.parse_args()

device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() and args.gpu >= 0 else "cpu")


with open('./data/gene_p.txt', 'r') as file:
    lines = file.readlines()
    i = 0
    for line in lines:
        if(i==0):
            gene_list = line.strip().split(' ')
        if(i==1):
            value = line.strip().split(' ')

        i += 1


value = list(map(float, value))
normalize_value = normalize_list(value)


data_feature=pd.read_csv('./data/L1000_Expr_subdata_lm_t.txt', sep='\t')
data_label=pd.read_excel('./data/L1000_Expr_subdata_sig_id_label.XLSX')['DILIst binaray classfication']
# x: gene feature(patients as rows and genes as columns )(n*p)
# y: label
x = data_feature.iloc[: , 1:].iloc[0: , :]
y = data_label

'''
x
            16        23        25        30        39        47       102  ...    147179    148022    200081    200734    256364    375346     388650
0    -0.273900  1.130100 -0.156750  0.687400 -4.058650  0.815700 -0.218150  ... -0.537950  2.260750 -1.621900 -0.832750 -0.534950  0.853400   1.815600
1    -2.592874 -1.112710 -0.562751  1.980042 -0.850379  1.009283  0.988296  ...  2.363817 -1.243268 -3.171911 -0.075645 -2.564447  2.868700   0.303892
2    -0.159000 -0.683950 -1.927550  1.124000  0.230600  0.143950  0.113850  ... -0.067000 -1.610100 -1.036000 -0.367700  0.166800  1.244100   0.786450
3    -0.770650  3.485550 -4.199150 -2.880950 -5.549300 -2.692650 -2.703900  ... -2.283850 -0.235100 -5.002700 -0.288700 -0.557150  0.899650   1.713000
4     1.181900  1.466400 -0.834100 -0.398200  2.232000 -2.027300  1.848400  ... -3.111400  2.705500 -3.419400 -7.587300 -0.942600  1.360400   1.473200
...        ...       ...       ...       ...       ...       ...       ...  ...       ...       ...       ...       ...       ...       ...        ...
5995  1.040600  0.067500  0.384300 -1.393800 -0.253800 -0.660000 -0.162300  ... -0.145300 -2.501300 -2.808700  1.790900 -0.678200 -0.014500 -10.000000
5996  1.452775 -0.174853  0.472281  0.210708  0.736352  0.101806 -0.022184  ...  0.252130 -1.072445  1.252616  0.299790 -0.412713  0.099130  -0.193858
5997 -3.063756  0.287776 -0.213062 -0.865963 -2.259327  2.488603  7.281178  ...  2.196773  1.963336  1.136331 -3.718022  2.555697  2.589732  -1.465106
5998  0.880278  0.175041 -0.312804  0.855141 -0.329998 -1.239330 -0.401126  ... -0.123923 -0.453814 -0.623671 -0.409342  0.266684  0.688531  -1.957750
5999 -0.045258  1.652625  1.524762 -0.032379 -0.746086  1.326783  1.469654  ...  0.542418  0.616446 -1.526584 -0.951404  0.440162  1.982720   2.464712

[6000 rows x 978 columns]

y
0       0
1       0
2       0
3       1
4       0
       ..
5995    1
5996    0
5997    1
5998    1
5999    1
Name: DILIst binaray classfication, Length: 6000, dtype: int64

'''
# ppi
ppi_matrix = pd.read_csv('./data/gene_978_matrix_700.csv', header=None) # 978个基因的关系矩阵
adj, edge = preprocess_Finaladj(ppi_matrix)
edge = torch.from_numpy(edge).to(device)
'''
adj:  (array([[  0,   0],
       [  0, 141],
       [  0, 764],
       ...,
       [975, 975],
       [976, 976],
       [977, 977]], dtype=int32), array([1.   , 0.988, 0.829, ..., 1.   , 1.   , 1.   ]), (978, 978))
edge:  [[  0   0   0 ... 975 976 977]
 [  0 141 764 ... 975 976 977]]
'''
def train_model(model, optimizer1, optimizer2, scheduler1, scheduler2, graphLearn_input, train_input, train_labels, test_input, test_labels, k_num, epochs=800):
    # Train model
    loss_list = []
    train_acc_list = []
    test_loss_list = []
    test_accuracy_list = []
    roc_auc_list = []


    for epoch in range(1, epochs+1):
        
        model.train()

        outputs = model(graphLearn_input, train_input)
        
        pred_labels = torch.argmax(outputs, dim=1).float().detach().cpu().numpy()
        
        true_labels = torch.argmax(train_labels, dim=1).float().detach().cpu().numpy()
        
        with torch.autograd.set_detect_anomaly(True):
            gl_loss_fn = glLoss(model, args.weight_decay, args.L1_gamma, args.L1_beta).to(device)
            gcn_loss_fn = gcnLoss(model, args.weight_decay).to(device)

            optimizer1.zero_grad()
            optimizer2.zero_grad()

            gl_loss = gl_loss_fn()
            gcn_loss = gcn_loss_fn(outputs, train_labels)

            accuracy = accuracy_score(true_labels, pred_labels)

            print("Training epoch: {} gl_loss: {} gcn_loss : {} accuracy: {}".format(epoch, gl_loss, gcn_loss, accuracy))

            loss_list.append(gcn_loss.detach().cpu())
            train_acc_list.append(accuracy)

            total_loss = gl_loss + 2*gcn_loss;
            total_loss.backward();

            # gl_loss.backward(retain_graph=True)
            # gcn_loss.backward()
            optimizer1.step()
            optimizer2.step()
            scheduler1.step()
            scheduler2.step()

            model.eval()
            test_outputs = model(graphLearn_input, test_input)

            # Converts the output of the model into the predicted category
            test_pred_labels = torch.argmax(test_outputs, dim=1).float().detach().cpu().numpy()
            # Converts real tags from one-hot encoding to category index
            test_true_labels = torch.argmax(test_labels, dim=1).float().detach().cpu().numpy()

            test_loss = torch.nn.CrossEntropyLoss()
            test_gcn_loss = test_loss(test_outputs, test_labels)
            test_loss_list.append(test_gcn_loss.detach().cpu())
            test_accuracy = accuracy_score(test_true_labels, test_pred_labels)
            test_pred = test_outputs[:, 1]
            fpr, tpr, thresholds = roc_curve(test_true_labels, test_pred.cpu().detach().numpy(), pos_label=1)
            
            precision, recall, pr_thresholds = precision_recall_curve(test_true_labels, test_pred.cpu().detach().numpy(), pos_label=1)
            
            roc_auc = auc(fpr, tpr)
            test_confusion_matrix = confusion_matrix(test_true_labels, test_pred_labels)
            TN, FP, FN, TP = test_confusion_matrix.ravel()
            Sensitivity = TP / (TP + FN)
            Specificity = TN / (TN + FP)
            roc_auc_list.append(roc_auc)
            test_accuracy_list.append(test_accuracy)

        acc_win = 'Accuracy-'+ str(k_num)
        loss_win = 'Loss-'+ str(k_num)
        sen_spec_win = 'Sen_Spec-'+ str(k_num)
        if epoch == 1:
            viz.line(
                [[gcn_loss.detach().cpu(), test_gcn_loss.detach().cpu()]],
                [epoch],
                win=loss_win,
                opts=dict(title=loss_win,
                            legend=['train_gcn_loss', 'test_gcn_loss']
                            )
            )
            viz.line(
                [[accuracy, test_accuracy]],
                [epoch],
                win=acc_win,
                opts=dict(title=acc_win,
                            legend=['train_acc', 'test_acc']
                            )
            )
            viz.line(
                [[Sensitivity, Specificity]],
                [epoch],
                win=sen_spec_win,
                opts=dict(title=sen_spec_win,
                            legend=['test_Sensitivity', 'test_Specificity']
                            )
            )
        else:
            viz.line(
                [[gcn_loss.detach().cpu(), test_gcn_loss.detach().cpu()]],
                [epoch],
                win=loss_win,
                update='append'
            )
            viz.line(
                [[accuracy, test_accuracy]],
                [epoch],
                win=acc_win,
                update='append'
            )
            viz.line(
                [[Sensitivity, Specificity]],
                [epoch],
                win=sen_spec_win,
                update='append'
            )

    return model

def test_model(model, graphLearn_input, test_input, test_labels, k_num):

    model.eval()
    test_outputs = model(graphLearn_input, test_input)

    # Converts the output of the model into the predicted category
    test_pred_labels = torch.argmax(test_outputs, dim=1).float().detach().cpu().numpy()
    # Converts real tags from one-hot encoding to category index
    test_true_labels = torch.argmax(test_labels, dim=1).float().detach().cpu().numpy()

    extracted_feature_flattened = model.flattened
    dense1 = model.lin1(extracted_feature_flattened)
    dense1 = model.BatchNorm1d_2(dense1)
    dense1 = model.relu(dense1)
    dense1 = torch.nn.Dropout(p=0.3)(dense1)

    dense2 = model.lin2(dense1)
    dense2 = model.BatchNorm1d_3(dense2)
    dense2 = model.relu(dense2)
    dense2 = torch.nn.Dropout(p=0.3)(dense2)
        
    dense3 = model.lin3(dense2)
    
    gcn_loss_fn = gcnLoss(model, args.weight_decay)
    test_gcn_loss = gcn_loss_fn(test_outputs, test_labels)

    test_accuracy = accuracy_score(test_true_labels, test_pred_labels)
    test_precision_score = precision_score(test_true_labels, test_pred_labels)
    test_recall_score = recall_score(test_true_labels, test_pred_labels)
    test_f1_score = f1_score(test_true_labels, test_pred_labels)
    
    test_pred = test_outputs[:, 1]
    fpr, tpr, thresholds = roc_curve(test_true_labels, test_pred.detach().cpu().numpy(), pos_label=1)

    test_confusion_matrix = confusion_matrix(test_true_labels, test_pred_labels)
    
    test_roc_auc = auc(fpr, tpr)
    test_confusion_matrix = confusion_matrix(test_true_labels, test_pred_labels)
    TN, FP, FN, TP = test_confusion_matrix.ravel()
    Sensitivity = TP / (TP + FN)
    Specificity = TN / (TN + FP)

    print("K-{} test loss: {} test accuracy: {} test_roc_auc: {} Sensitivity: {} Specificity: {}".format(k_num, test_gcn_loss, test_accuracy, test_roc_auc, Sensitivity, Specificity))

    return test_gcn_loss, test_accuracy, test_precision_score, test_recall_score, test_f1_score, fpr, tpr, thresholds, test_confusion_matrix



k_num = 0
train_log = []
best_accuracy = 0.0
        
for train_index, test_index in KF.split(x,y):

    print("-------------------------No.{} fold-Cross-Validation-------------------------".format(k_num))

    train_mask = sample_mask(train_index, y.shape[0])
    test_mask = sample_mask(test_index, y.shape[0])

    y_train = y[train_mask].reset_index(drop=True)  # train label
    y_test = y[test_mask].reset_index(drop=True)  # test label

    # --------------------train dataset------------------------------#
    input = x[train_mask].reset_index(drop=True) # get train dataset through train_mask

    # Obtain data with rows of genes and columns of drug and normalize p*1
    graphLearn_input = preprocess_features(np.transpose(input))
    
    second = np.array(input)
    # Extend dimension to n*p*1
    train_input = np.expand_dims(second, -1) # train_input
    train_input = torch.from_numpy(train_input).to(device)
    
    train_labels = torch.nn.functional.one_hot(torch.tensor(y_train.values))
    train_labels = torch.Tensor(train_labels).to(device).float()

    #--------------------test dataset------------------------------#
    test_input = x[test_mask].reset_index(drop=True)
    test_input = np.array(test_input)
    # Extend dimension to n*p*1
    test_input = np.expand_dims(test_input, -1)
    test_input = torch.from_numpy(test_input).to(device)

    test_labels = torch.nn.functional.one_hot(torch.tensor(y_test))
    test_labels = torch.Tensor(test_labels).to(device).float()

    # init model
    model = BioGLGCN(edge=edge,
            adj=adj,
            gene_p=normalize_value,
            num=graphLearn_input[2][0],
            input_dim=graphLearn_input[2][1],
            output_dim=2,
            phi=args.Phi,
            layer0_outputdim=args.layer0_outputdim,
            layer1_outputdim=args.layer1_outputdim,
            layer2_outputdim=args.layer2_outputdim,
            dropout=args.dropout,
            ).to(device)

    # init optimizer,lr
    # GL optimizer
    optimizer1 = torch.optim.Adam(model.layers0.parameters(), lr=0.001)
    # GCN, Linear optimizer
    parameters_to_optimize = list(model.layers1.parameters()) + list(model.lin1.parameters()) + list(model.lin2.parameters()) + list(model.lin3.parameters())

    optimizer2 = torch.optim.Adam(parameters_to_optimize, lr=0.00002)

    # lr update rule
    scheduler1 = torch.optim.lr_scheduler.StepLR(optimizer1, step_size=100, gamma=args.gamma)
    scheduler2 = torch.optim.lr_scheduler.StepLR(optimizer2, step_size=50, gamma=args.gamma)


    #--------------------------------train model------------------------------#
    train_epoch = args.epoch
    model_conv = train_model(model, optimizer1, optimizer2, scheduler1, scheduler2, graphLearn_input, train_input, train_labels, test_input, test_labels, k_num, epochs=train_epoch)
    # 保存每一折的特征
    
    #--------------------------------test model------------------------------#
    test_loss, test_accuracy, test_precision_score, test_recall_score, test_f1_score, fpr, tpr, thresholds, test_confusion_matrix= test_model(model_conv, graphLearn_input, test_input, test_labels, k_num)

    if test_accuracy > best_accuracy:
        best_accuracy = test_accuracy
        best_model_state = model.state_dict()


    log_dict = [k_num, test_loss, test_accuracy, test_precision_score, test_recall_score, test_f1_score, fpr, tpr, thresholds, test_confusion_matrix]
    train_log.append(log_dict)

    k_num += 1



tprs=[]
aucs=[]
mean_fpr=np.linspace(0,1,100)

with open('./results/trainlog/trainlog.txt', 'w', encoding='utf8') as f:
    sum_acc = sum_prec = sum_recall = sum_f1 = sum_mcc = sum_Sensitivity = sum_Specificity = sum_auc = 0.0
    k = 0
    for j in train_log:
        TP = j[9][1, 1]
        FP = j[9][0, 1]
        TN = j[9][0, 0]
        FN = j[9][1, 0]
        fpr = j[6]
        tpr = j[7]
        roc_auc = auc(fpr, tpr)

        tprs.append(np.interp(mean_fpr,fpr,tpr))
        tprs[-1][0]=0.0
        aucs.append(roc_auc)

        with open('./results/roc_pickle/BioGL-GCN_roc_data.pickle', 'wb') as plot_f:
            pickle.dump({'tprs': tprs, 'aucs': aucs, 'mean_fpr': mean_fpr}, plot_f)

        Sensitivity = TP / (TP + FN)
        Specificity = TN / (TN + FP)

        mcc = ((TP * TN) - (FP * FN)) / (math.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN)))

        f.write('No.{} fold-Cross-Validation acc: {} prec: {} recall: {} f1: {} MCC: {} Sensitivity: {} Specificity: {} auc: {}\n'.format(j[0], j[2], j[3], j[4], j[5], mcc, Sensitivity, Specificity, roc_auc))
        
        sum_acc += float(j[2])
        sum_prec += float(j[3])
        sum_recall += float(j[4])
        sum_f1 += float(j[5])
        sum_mcc += float(mcc)
        sum_Sensitivity += float(Sensitivity)
        sum_Specificity += float(Specificity)
        sum_auc += float(roc_auc)

        plt.clf()
        plt.title('Test ROC')
        plt.plot(fpr, tpr, 'b', label = 'Test AUC = %0.3f' % roc_auc)
        plt.legend(loc = 'lower right')
        plt.plot([0, 1], [0, 1],'r--')
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        plt.savefig('./results/roc_figure/roc{}.png'.format(k))
        k += 1
    
    avg_acc = sum_acc / k_num   # k_num = 5
    avg_prec = sum_prec / k_num
    avg_recall = sum_recall / k_num
    avg_f1 = sum_f1 / k_num
    avg_mcc = sum_mcc / k_num
    avg_Sensitivity = sum_Sensitivity / k_num
    avg_Specificity = sum_Specificity / k_num
    avg_auc = sum_auc / k_num
    print(k_num)
    

    print('Avg acc: {} prec: {} reacll: {} f1:{} mcc: {} Sensitivity:{} Specificity: {} auc: {}'.format(avg_acc, avg_prec, avg_recall, avg_f1, avg_mcc, avg_Sensitivity, avg_Specificity, avg_auc))
    f.write('Avg acc: {} prec: {} reacll: {} f1:{} Sensitivity:{} Specificity: {} auc: {}\n'.format(avg_acc, avg_prec, avg_recall, avg_f1, avg_Sensitivity, avg_Specificity, avg_auc))

torch.save(best_model_state, './results/model/978_BioGL-GCN_best_model_test.pth')