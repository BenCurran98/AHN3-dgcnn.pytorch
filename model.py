#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: Yue Wang
@Contact: yuewangx@mit.edu
@File: model.py
@Time: 2018/10/13 6:35 PM

Modified by 
@Author: An Tao
@Contact: ta19@mails.tsinghua.edu.cn
@Time: 2020/3/9 9:32 PM
"""


import os
import sys
import copy
import math
import numpy as np
import torch
import torch.nn as nn


def knn(x, k):
    """Calculate the k Nearest Neighbours for each point in a feature tensor

    Args:
        x (tensor): Feature tensor to have neighbours calculated for
        k (int): Number of neighbours to find

    Returns:
        idx: List of kNN indices for each point in the feature matrix
    """
    inner = -2*torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
 
    idx = pairwise_distance.topk(k=k, dim=-1)[1]   # (batch_size, num_points, k)
    return idx


def get_graph_feature(x, k=20, use_cuda=True):
    """Perform a spatial graph convolution on a feature matrix

    Args:
        x (tensor): Feature tensor being processed
        k (int, optional): Number of neighbours to find. Defaults to 20.
        use_cuda (bool, optional): Whether to send data to the GPU. Defaults to True.

    Returns:
        feature (tensor): Feature tensor extracted from the graph
    """
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    idx = knn(x, k=k)   # (batch_size, num_points, k)

    if use_cuda:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1)*num_points
    idx = idx.to(device)

    idx = idx + idx_base

    idx = idx.view(-1)
 
    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous()   # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    feature = x.view(batch_size*num_points, -1)[idx, :]

    
    feature = feature.view(batch_size, num_points, k, num_dims) 
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)

    # now join the global features (x) and the local features (feature - x)
    feature = torch.cat((feature-x, x), dim=3).permute(0, 3, 1, 2)
  
    return feature      # (batch_size, 2*num_dims, num_points, k)

class DGCNN(nn.Module):
    """A class used to represent a Dynamic Graph Convolutional Neural Network (DGCNN)
    """
    def __init__(self, num_classes, num_features, k, 
                    dropout = 0.5, 
                    emb_dims = 1024, 
                    cuda = False):
        """
        Args:
            num_classes (int): Number of classes for the model to learn
            num_features (int): Number of point cloud features for the model to learn on
            k (int): Number of neighbours to search for in EdgeConv layers
            dropout (float, optional): Probability of dropout in model. Defaults to 0.5.
            emb_dims (int, optional): Dimension where features are embedded into in the network. Defaults to 1024.
            cuda (bool, optional): Whether to send model to GPU. Defaults to False.
        """
        super(DGCNN, self).__init__()
        self.k = k
        self.num_classes = num_classes
        self.use_cuda = cuda
        self.num_features = num_features
        
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(64)
        self.bn4 = nn.BatchNorm2d(64)
        self.bn5 = nn.BatchNorm2d(64)
        self.bn6 = nn.BatchNorm1d(emb_dims)
        self.bn7 = nn.BatchNorm1d(512)
        self.bn8 = nn.BatchNorm1d(256)

        self.conv1 = nn.Sequential(nn.Conv2d(2 * num_features, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(64*2, 64, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=False),
                                   self.bn4,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv5 = nn.Sequential(nn.Conv2d(64*2, 64, kernel_size=1, bias=False),
                                   self.bn5,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv6 = nn.Sequential(nn.Conv1d(192, emb_dims, kernel_size=1, bias=False),
                                   self.bn6,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv7 = nn.Sequential(nn.Conv1d(1216, 512, kernel_size=1, bias=False),
                                   self.bn7,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv8 = nn.Sequential(nn.Conv1d(512, 256, kernel_size=1, bias=False),
                                   self.bn8,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.dp1 = nn.Dropout(p=dropout)
        self.conv9 = nn.Conv1d(256, self.num_classes, kernel_size=1, bias=False)
        

    def forward(self, x, depth = 10):
        num_points = x.size(2)

        x = get_graph_feature(x, k=self.k, dim9=True, use_cuda=self.use_cuda)   # (batch_size, num_features, num_points) -> (batch_size, 2 * num_features, num_points, k)
        x = self.conv1(x)                       # (batch_size, 2 * num_features, num_points, k) -> (batch_size, 64, num_points, k)
        x = self.conv2(x)                       # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points, k)
        x1 = x.max(dim=-1, keepdim=False)[0]    # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)
        if depth == 1:
            return x1

        x = get_graph_feature(x1, k=self.k, use_cuda=self.use_cuda)     # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        x = self.conv3(x)                       # (batch_size, 64*2, num_points, k) -> (batch_size, 64, num_points, k)
        x = self.conv4(x)                       # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points, k)
        x2 = x.max(dim=-1, keepdim=False)[0]    # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)
        if depth == 2:
            return x2

        x = get_graph_feature(x2, k=self.k, use_cuda=self.use_cuda)     # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        x = self.conv5(x)                       # (batch_size, 64*2, num_points, k) -> (batch_size, 64, num_points, k)
        x3 = x.max(dim=-1, keepdim=False)[0]    # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)
        if depth == 3:
            return x3

        x = torch.cat((x1, x2, x3), dim=1)      # (batch_size, 64*3, num_points)
        if depth == 4:
                return x

        # this is where the features get embedded in a high-dimensional space
        x = self.conv6(x)                       # (batch_size, 64*3, num_points) -> (batch_size, emb_dims, num_points)
        if depth == 5:
                return x
        x = x.max(dim=-1, keepdim=True)[0]      # (batch_size, emb_dims, num_points) -> (batch_size, emb_dims, 1)
        if depth == 6:
                return x

        x = x.repeat(1, 1, num_points)          # (batch_size, emb_dims, num_points)
        x = torch.cat((x, x1, x2, x3), dim=1)   # (batch_size, emb_dims+64*3, num_points)

        if depth == 7:
                return x
        x = self.conv7(x)                       # (batch_size, emb_dims+64*3, num_points) -> (batch_size, 512, num_points)
        if depth == 8:
                return x
        x = self.conv8(x)                       # (batch_size, 512, num_points) -> (batch_size, 256, num_points)
        if depth == 9:
                return x
        x = self.dp1(x)
        x = self.conv9(x)                       # (batch_size, 256, num_points) -> (batch_size, num_classes, num_points)
        
        return x