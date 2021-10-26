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
from torch.jit import Error
from util import IOStream
from train import train_args
from test import test_args
from prepare_data.process_data import process_data


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
    parser = argparse.ArgumentParser(description='DGCNN Interface')
    AREA = 'Training'
    PC_DIR = os.path.join(os.getcwd(), '..' '/Datasets', 'QualityTraining-orig')
    BASE_DIR = os.path.join(os.getcwd(), '..' '/Datasets')
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
    CLASS_MAP_FILE = os.path.join(ROOT_DIR, "params", "class_map.json")

    # interfacey stuff
    parser.add_argument('--eval', type=bool, default=False, help='evaluate the model')
    parser.add_argument("--mode", type = str, default = "classifier", help = "What mode to execute (classifier or process_data)")

    # train/test related args go here
    parser.add_argument('--data_dir', type=str, default='/media/ben/ExtraStorage/InnovationConference/Datasets/data_as_S3DIS_NRI_NPY', help='Directory of data')
    parser.add_argument('--tb_dir', type=str, default='log_tensorboard', help='Directory of tensorboard logs')
    parser.add_argument('--exp_name', type=str, default='dgcnn_test_30epochs_p100', metavar='N', help='Name of the experiment')
    parser.add_argument('--model', type=str, default='dgcnn', metavar='N', choices=['dgcnn'], help='Model to use, [dgcnn]')
    parser.add_argument('--block_size', type=float, default=30.0, help='size of one block')
    parser.add_argument('--num_classes', type=int, default=5, help='number of classes in the dataset')
    parser.add_argument('--num_features', type=int, default=3, help='Number of pointcloud feature columns in data')
    parser.add_argument('--validation_area', type=str, default='4', metavar='N', choices=['1', '2', '3', '4', 'all'])
    parser.add_argument('--batch_size', type=int, default=12, metavar='batch_size', help='Size of batch)')
    parser.add_argument('--validation_batch_size', type=int, default=12, metavar='batch_size', help='Size of batch)')
    parser.add_argument('--epochs', type=int, default=50, metavar='N', help='number of episode to train ')
    parser.add_argument('--use_sgd', type=bool, default=False, help='Use SGD')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR', help='learning rate (default: 0.001, 0.1 if using sgd)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M', help='SGD momentum (default: 0.9)')
    parser.add_argument('--scheduler', type=str, default='cos', metavar='N', choices=['cos', 'step'], help='Scheduler to use, [cos, step]')
    parser.add_argument('--no_cuda', type=bool, default=False, help='enables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
    parser.add_argument('--dropout', type=float, default=0.5, help='dropout rate')
    parser.add_argument('--emb_dims', type=int, default=1024, metavar='N', help='Dimension of embeddings')
    parser.add_argument('--k', type=int, default=20, metavar='N', help='Num of nearest neighbors to use')
    parser.add_argument('--model_root', type=str, default='checkpoints/RGB_30m/models', metavar='N', help='Pretrained model root')
    parser.add_argument('--test_visu_dir', default='predict', help='Directory of test visualization files.')
    parser.add_argument('--validation_prop', type=float, default = 0.2, metavar = 'N', help = 'Proportion of data to use for testing')
    parser.add_argument('--sample_num', type=int, default = 5, metavar = 'N', help = 'Number of training blocks to randomly sample')
    parser.add_argument('--num_threads', type=int, default = 8, metavar = 'N', help = 'Number of threads to use for training')
    parser.add_argument('--num_interop_threads', type=int, default = 2, metavar = 'N', help = 'Number of threads to use for inter-operations in pytorch')
    parser.add_argument('--exclude_classes', nargs = "*", type=int, default = -1, metavar = 'N', help = 'Class labels to ignore in training')
    parser.add_argument('--min_class_num', type = int, default = 100, help = 'Minimum number of points per class for the pointcloud to be used')
    parser.add_argument('--model_label', type = str, default = "dgcnn_model", help = 'Label of model file')
    parser.add_argument('--min_class_confidence', type = float, default = 0.8, help = 'Minimum confidence value for the model to label a point as belonging to a class')

    # data preprocessing related args go here
    parser.add_argument('--base_dir', type = str, default = os.path.join(BASE_DIR, AREA), help = 'Base directory of data')
    parser.add_argument('--root_dir', type = str, default = ROOT_DIR, help = 'Root directory of the files')
    parser.add_argument('--area', type = str, default = AREA, help = 'Name of area to process')
    parser.add_argument('--pc_folder', type = str, default = PC_DIR)
    parser.add_argument('--processed_data_folder', type = str, default = os.path.join(BASE_DIR, AREA, "processed"), help = 'Folder containing the complete datasets')
    parser.add_argument('--categories_file', type = str, default = 'params/categories.json', help = 'JSON file containing label mappings')
    parser.add_argument('--features_file', type = str, default = 'params/features.json', help = 'JSON file containing index mappings of LiDAR features')
    parser.add_argument('--class_map_file', type = str, default = CLASS_MAP_FILE, help = 'File containing class mappings')
    parser.add_argument('--features_output', nargs = '*', type = str, default = ["x", "y", "z", "agl"], help = 'LiDAR features to extract')
    parser.add_argument('--npy_data_folder', type = str, default = os.path.join(BASE_DIR, 'data_as_S3DIS_NRI_NPY'), help = 'Output folder of the data summary')
    parser.add_argument('--calc_agl', type = bool, default = True, help = 'Whether to calculate AGL for the pointcloud')
    parser.add_argument('--cell_size', type = int, default = 1, help = 'Size of DTM cell')
    parser.add_argument('--desired_seed_cell_size', type = int, default = 90, help = 'Size of DTM seed cell')
    parser.add_argument('--boundary_block_width', type = int, default = 5, help = 'Number of blocks to use on the boundary')
    parser.add_argument('--detect_water', type = bool, default = False, help = 'Whether to detect water in DTM generation')
    parser.add_argument('--remove_buildings', type = bool, default = True, help = 'Whether to remove buildings in DTM generation')
    parser.add_argument('--output_tin_file_path', type = any, default = None, help = 'File path of the DTM tin file to produce')
    parser.add_argument('--dtm_buffer', type = float, default = 6, help = 'Buffer (metres) around the DTM region to use')
    parser.add_argument('--dtm_module_path', type = str, default = "/media/ben/ExtraStorage/external/RoamesDtmGenerator/bin", help = 'Path to the RoamesDTMGenerator module')
    parser.add_argument('--num_points', type = int, default = 7000, help = 'Number of points to subsample from each sub tile')
    parser.add_argument('--sub_block_size', type = float, default = 30, help = 'Size of sub blocks that each tile is broken into')
    parser.add_argument('--use_all_points', type = bool, default = False, help = 'Whether or not to use all points in each sub block')
    parser.add_argument('--sub_sample_num', type = int, default = 5, help = 'Number of sub tile samples to take from each tile')
    parser.add_argument('--n_tries', type = int, default = 10, help = 'Number of searches to perform for suitable sub tiles')
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

    if args.mode == "classifier":
        if not args.eval:
            train_args(args, io)
        else:
            test_args(args, io)
    elif args.mode == "process_data":
        process_data(args.base_dir, args.root_dir, args.pc_folder, args.data_folder, 
                args.processed_data_folder, args.npy_data_folder, args.area, 
                args.categories_file, args.features_file, args.features_output, 
                args.block_size, args.sample_num, args.min_class_num, args.class_map_file,
                args.calc_agl, args.cell_size, args.desired_seed_cell_size,
                args.boundary_block_width, args.detect_water,
                args.remove_buildings, args.output_tin_file_path, 
                args.dtm_buffer, args.dtm_module_path, args.num_points,
                args.sub_block_size, args.use_all_points, args.sub_sample_num,
                args.n_tries)
    else:
        raise(Error("Invalid operation mode entered!"))
