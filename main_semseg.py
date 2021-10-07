#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: An Tao
@Contact: ta19@mails.tsinghua.edu.cn
@File: main_semseg.py
@Time: 2020/2/24 7:17 PM
"""

from __future__ import print_function
import os
import argparse
import torch
from util import IOStream
from train import train_args
from test import test_args


def _init_():
    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')
    if not os.path.exists('checkpoints/' + args.exp_name):
        os.makedirs('checkpoints/' + args.exp_name)
    if not os.path.exists('checkpoints/' + args.exp_name + '/' + 'models'):
        os.makedirs('checkpoints/' + args.exp_name + '/' + 'models')
    os.system('cp main_semseg.py checkpoints' + '/' + args.exp_name + '/' + 'main_semseg.py.backup')
    os.system('cp model.py checkpoints' + '/' + args.exp_name + '/' + 'model.py.backup')
    os.system('cp util.py checkpoints' + '/' + args.exp_name + '/' + 'util.py.backup')
    os.system('cp data.py checkpoints' + '/' + args.exp_name + '/' + 'data.py.backup')

if __name__ == "__main__":
    # Training settings
    parser = argparse.ArgumentParser(description='Point Cloud Semantic Segmentation')
    parser.add_argument('--data_dir', type=str, default='/media/ben/ExtraStorage/InnovationConference/Datasets/data_as_S3DIS_NRI_NPY',
                        help='Directory of data')
    parser.add_argument('--tb_dir', type=str, default='log_tensorboard',
                        help='Directory of tensorboard logs')
    parser.add_argument('--exp_name', type=str, default='dgcnn_test_30epochs_p100', metavar='N',
                        help='Name of the experiment')
    parser.add_argument('--model', type=str, default='dgcnn', metavar='N',
                        choices=['dgcnn'],
                        help='Model to use, [dgcnn]')
    parser.add_argument('--dataset', type=str, default='S3DIS', metavar='N',
                        choices=['S3DIS'])
    parser.add_argument('--block_size', type=float, default=30.0,
                        help='size of one block')
    parser.add_argument('--num_classes', type=int, default=5,
                        help='number of classes in the dataset')
    parser.add_argument('--num_features', type=int, default=3,
                        help='Number of pointcloud feature columns in data')
    parser.add_argument('--test_area', type=str, default='4', metavar='N',
                        choices=['1', '2', '3', '4', 'all'])
    parser.add_argument('--batch_size', type=int, default=12, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--test_batch_size', type=int, default=12, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--epochs', type=int, default=50, metavar='N',
                        help='number of episode to train ')
    parser.add_argument('--use_all_points', type=bool, default=False, metavar='N',
                        help='Whether to use all points in block')
    parser.add_argument('--use_sgd', type=bool, default=False,
                        help='Use SGD')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001, 0.1 if using sgd)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--scheduler', type=str, default='cos', metavar='N',
                        choices=['cos', 'step'],
                        help='Scheduler to use, [cos, step]')
    parser.add_argument('--no_cuda', type=bool, default=False,
                        help='enables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--eval', type=bool, default=False,
                        help='evaluate the model')
    parser.add_argument('--num_points', type=int, default=4096,
                        help='num of points to use')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='dropout rate')
    parser.add_argument('--emb_dims', type=int, default=1024, metavar='N',
                        help='Dimension of embeddings')
    parser.add_argument('--k', type=int, default=20, metavar='N',
                        help='Num of nearest neighbors to use')
    parser.add_argument('--model_root', type=str, default='checkpoints/RGB_30m/models', metavar='N',
                        help='Pretrained model root')
    parser.add_argument('--test_visu_dir', default='predict',
                        help='Directory of test visualization files.')
    parser.add_argument('--test_prop', type=float, default = 0.2, metavar = 'N',
                        help = 'Proportion of data to use for testing')
    parser.add_argument('--sample_num', type=int, default = 5, metavar = 'N',
                        help = 'Number of training blocks to randomly sample')
    parser.add_argument('--num_threads', type=int, default = 8, metavar = 'N',
                        help = 'Number of threads to use for training')
    parser.add_argument('--num_interop_threads', type=int, default = 2, metavar = 'N',
                        help = 'Number of threads to use for inter-operations in pytorch')
    parser.add_argument('--exclude_classes', nargs = "*", type=int, default = -1, metavar = 'N',
                        help = 'Class labels to ignore in training')
    parser.add_argument('--min_class_num', type = int, default = 100, 
                        help = 'Minimum number of points per class for the pointcloud to be used')
    parser.add_argument('--model_label', type = str, default = "dgcnn_model", 
                        help = 'Label of model file')
    parser.add_argument('--min_class_confidence', type = float, default = 0.8, 
                        help = 'Minimum confidence value for the model to label a point as belonging to a class')
    args = parser.parse_args()

    _init_()

    io = IOStream('checkpoints/' + args.exp_name + '/run.log')
    io.cprint(str(args))

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    if args.cuda:
        io.cprint(
            'Using GPU : ' + str(torch.cuda.current_device()) + ' from ' + str(torch.cuda.device_count()) + ' devices')
        torch.cuda.manual_seed(args.seed)
    else:
        io.cprint('Using CPU')

    if not args.eval:
        train_args(args, io)
    else:
        test_args(args, io)
