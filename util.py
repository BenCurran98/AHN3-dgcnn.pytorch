#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: Yue Wang
@Contact: yuewangx@mit.edu
@File: util
@Time: 4/5/19 3:47 PM
"""


import numpy as np
import torch
import torch.nn.functional as F
from prettytable import PrettyTable


def cal_loss(pred, gold, smoothing=True):
    ''' Calculate cross entropy loss, apply label smoothing if needed. '''

    gold = gold.contiguous().view(-1)

    if smoothing:
        eps = 0.2
        n_class = pred.size(1)

        one_hot = torch.zeros_like(pred).scatter(1, gold.view(-1, 1), 1)
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
        log_prb = F.log_softmax(pred, dim=1)

        loss = -(one_hot * log_prb).sum(dim=1).mean()
    else:
        loss = F.cross_entropy(pred, gold, reduction='mean')

    return loss


class IOStream():
    """A class for streaming data
    """
    def __init__(self, path):
        """
        Args:
            path (str): Path to file to stream data into
        """
        self.f = open(path, 'a')

    def cprint(self, text):
        """Print data into the IO file

        Args:
            text (str): Text data to be written to file
        """
        print(text)
        self.f.write(text+'\n')
        self.f.flush()

    def close(self):
        self.f.close()

def calculate_sem_IoU(pred_np, seg_np, num_classes):
    """Calculate the Intersection Over Union of the predicted classes and the ground truth

    Args:
        pred_np (array_like): List of predicted class labels
        seg_np (array_like): List of ground truth labels
        num_classes (int): Number of classes in the dataset
    """
    I_all = np.zeros(num_classes)
    U_all = np.zeros(num_classes)
    for sem_idx in range(len(seg_np)):
        for sem in range(num_classes):
            I = np.sum(np.logical_and(pred_np[sem_idx] == sem, seg_np[sem_idx] == sem))
            U = np.sum(np.logical_or(pred_np[sem_idx] == sem, seg_np[sem_idx] == sem))
            I_all[sem] += I
            U_all[sem] += U
    return I_all / U_all

def count_parameters(model):
    """Count the numbeer of total and trainable parameters in a model"""
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: 
            continue
        param = parameter.numel()
        table.add_row([name, param])
        total_params+=param
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params