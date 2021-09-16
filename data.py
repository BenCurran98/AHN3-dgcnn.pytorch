#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: Yue Wang
@Contact: yuewangx@mit.edu
@File: data.py
@Time: 2018/10/13 6:21 PM

Modified by 
@Author: An Tao
@Contact: ta19@mails.tsinghua.edu.cn
@Time: 2020/2/27 9:32 PM
"""


import os
import sys
import glob
import threading
import h5py
import laspy
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
import prepare_data.pointcloud_util as util

class FugroDataset(Dataset):
    """
    A class to wrap around a Fugro point cloud dataset

    ...
    Attributes
    ----------
    split : str
        Indicates if this is a training or test data set
    data_root : str
        Default location of the directory containing the data (default '')
    num_point : int
        The number of points to sample from a point cloud block (default 4096)
    block_size : float
        Size of the blocks to subdivide a tile into (default 30.0)
    use_all_points : bool
        Whether to use all points in a block or to subsample (default False)
    test_prop : float
        Fraction of data set to use as test/validation data (default 0.2)
    classes : list
        List of classes used in the data set (default [1, 2, 3, 4, 5])
    sample_num : int
        Number of blocks to randomly sample from each tile (default 5)
    """
    def __init__(self, split='train', data_root='', num_point=4096, block_size=30.0, use_all_points=False, test_prop = 0.2, classes = [0, 1, 2, 3, 4], sample_num = 5, class_min = 100):
        super().__init__()
        self.num_point = num_point
        self.block_size = block_size
        self.use_all_points = use_all_points
        self.test_prop = test_prop
        self.classes = classes
        rooms = sorted(os.listdir(data_root))
        rooms = [room for room in rooms if 'Area_' in room]

        test_areas = np.random.choice(range(len(rooms)), int(np.floor(len(rooms) * self.test_prop)), replace = False)
        if split == "train":
            rooms_split = [room for room in rooms if not any(['Area_{}'.format(test_area) in room for test_area in test_areas])]
        else:
            rooms_split = [rooms[i] for i in test_areas]
            
        self.room_points, self.room_labels = [], []
        self.room_coord_min, self.room_coord_max = [], []

        room_idxs = []
        num_point_all = []
        for index in tqdm(range(len(rooms_split)), "Sampling Tiles"):
            room_name = rooms_split[index]
            room_path = os.path.join(data_root, room_name)
            room_data = np.load(room_path)
            points, labels = room_data[:, 0:3], room_data[:, 3]
            coord_min, coord_max = np.amin(points, axis=0)[:3], np.amax(points, axis=0)[:3]
            self.room_coord_min.append(coord_min), self.room_coord_max.append(coord_max)

            las = laspy.create(file_version = "1.2", point_format = 3) 

            las.x = points[:, 0]
            las.y = points[:, 1]
            las.z = points[:, 2]

            las.classification = labels

            las.write("../DataSampleTrain/{}_block_data{}.las".format(split, index))

            if split == "train": 
                found = 0
                while found < sample_num:
                    block_points, block_labels = util.room2blocks(points, labels, self.num_point, block_size=self.block_size,
                                                            stride=self.block_size/3, random_sample=True, sample_num=sample_num - found, use_all_points=self.use_all_points)
            
                    for i in range(block_points.shape[0]):
                        this_block_points = block_points[i, :, :]
                        this_block_labels = block_labels[i, :]
                        label_counts = [len(np.where(this_block_labels == c)[0]) for c in classes]
                        # print("Label counts: ", label_counts)
                        if all([c > class_min for c in label_counts]):
                            # print("FOUND ONE: ", found)
                            found += 1
                            room_idxs.extend([index])
                            self.room_points.append(np.reshape(this_block_points, (1, this_block_points.shape[0], this_block_points.shape[1])))
                            self.room_labels.append(np.reshape(this_block_labels, (1, this_block_labels.shape[0])))
                            num_point_all.append(this_block_labels.size)

                las = laspy.create(file_version = "1.2", point_format = 3) 

                las.x = self.room_points[-1][0, :, 0]
                las.y = self.room_points[-1][0, :, 1]
                las.z = self.room_points[-1][0, :, 2]

                las.classification = self.room_labels[-1][0, :]

                las.write("../DataSampleTrain/{}_subsampled__block_data{}.las".format(split, index))

            else:
                block_points, block_labels = util.room2blocks(points, labels, self.num_point, block_size=self.block_size,
                                                            stride=self.block_size/3, random_sample=True, sample_num=sample_num, use_all_points=self.use_all_points)
                room_idxs.extend([index] * int(block_points.shape[0]))  # extend with number of blocks in a room
                self.room_points.append(block_points), self.room_labels.append(block_labels)
                num_point_all.append(labels.size)
        self.room_points = np.concatenate(self.room_points)
        self.room_labels = np.concatenate(self.room_labels)

        self.room_idxs = np.array(room_idxs)

        print("Totally {} samples in {} set.".format(len(self.room_idxs), split))

    def create_train_mask(self, idx, tot_samples):
        """Generate a binary mask to select training points to contribute to learning

        Args:
            idx (int): Index of the labels in the data set that are having the mask applied
            tot_samples (int): Total number of selected points we would like

        Returns:
            train_mask: Binary mask where a 1 indicates that this label will be used in back propogation
        """
        labels = self.room_labels[idx]
        label_counts = np.zeros(len(self.classes))
        for i in range(len(self.classes)):
            label_counts[i] = np.sum(labels == self.classes[i])
        min_label_count = min([count for count in label_counts if count > 0])
        n_samples = int(min(min_label_count, np.floor(tot_samples/len(self.classes))))
        train_mask = np.zeros(labels.shape)
        for label in self.classes:
            this_label_idxs = np.where(labels == label)[0]
            if len(this_label_idxs) > 0:
                training_idxs = np.random.choice(this_label_idxs, n_samples, replace = False)
                train_mask[training_idxs] = 1
        return train_mask

    def sample_points(self, idx, tot_samples):
        """Sample points from a tile

        Args:
            idx (int): Index of the tile in the data set we are sampling from
            tot_samples (int): Total number of points to sample

        Returns:
            selected_point_idxs: List of point indexs that we sample
        """
        labels = self.room_labels[idx]
        label_counts = np.zeros(len(self.classes))
        for i in range(len(self.classes)):
            label_counts[i] = np.sum(labels == self.classes[i])
            
        tot = sum(label_counts)
        shuf_nums = np.array([tot - i for i in label_counts])
        tot = sum(shuf_nums)
        weights = shuf_nums/tot
        weights.astype('float64')
        weights /= weights.sum()

        point_weights = np.zeros(len(labels))
        for i in range(len(labels)):
            point_weights[i] = weights[int(labels[i])]
        point_weights /= point_weights.sum()
        if np.isnan(sum(point_weights)):
            point_weights = [1/len(labels) for i in range(len(labels))]

        point_idxs = [i for i in range(len(labels))]

        # randomly select indices, weighted inversely proportional to the number of points of this type
        selected_point_idxs = np.random.choice(point_idxs, tot_samples, p=point_weights, replace=False)

        return selected_point_idxs


    def __getitem__(self, idx):  # get items in one block
        room_idx = self.room_idxs[idx]
        points = self.room_points[room_idx]   # N * 3

        labels = self.room_labels[room_idx]   # N

        N_points = points.shape[0]

        if not self.use_all_points:
            N_points = self.num_point

        # subsample the points
        selected_point_idxs = self.sample_points(idx, N_points)

        selected_points = points[selected_point_idxs, :]
        selected_labels = labels[selected_point_idxs]

        # generate a binary training mask for the points (which will be ignored in testing)
        mask = self.create_train_mask(idx, N_points)
        
        return selected_points, selected_labels, mask

    def __len__(self):
        return len(self.room_idxs)

class FugroDataset_eval(Dataset):
    def __init__(self, split='train', data_root='', num_point=4096, block_size=30.0, use_all_points=False):
        super().__init__()
        self.num_point = num_point
        self.block_size = block_size
        
        self.use_all_points = use_all_points
        rooms = sorted(os.listdir(data_root))
        rooms_split = [room for room in rooms if 'Area_' in room]
        
        self.room_points, self.room_labels = [], []
        self.room_coord_min, self.room_coord_max = [], []
        self.centres = []

        room_idxs = []
        for index in tqdm(range(len(rooms_split)), "Sampling Tiles"):
            room_name = rooms_split[index]
            room_path = os.path.join(data_root, room_name)
            room_path = os.path.join(data_root, room_name)
            room_data = np.load(room_path)
            points, labels = room_data[:, 0:3], room_data[:, 3]
            coord_min, coord_max = np.amin(points, axis=0)[:3], np.amax(points, axis=0)[:3]
            self.room_coord_min.append(coord_min), self.room_coord_max.append(coord_max)
            block_points, block_labels = util.room2blocks(points, labels, self.num_point, block_size=self.block_size,
                                                       stride=self.block_size, random_sample=False, sample_num=None, use_all_points=self.use_all_points)
            these_centres = []
            for i in range(block_points.shape[0]):
                this_centre = np.mean(block_points[i, :])
                these_centres.append(this_centre)
            f = open("../DataSampleTest/block_data{}.txt".format(index), "w")
            these_points = np.concatenate(block_points)
            for i in range(these_points.shape[0]):
                f.write("%f %f %f\n" % (these_points[i, 0], these_points[i, 1], these_points[i, 2]))
            f.close()
            room_idxs.extend([index] * int(block_points.shape[0]))  # extend with number of blocks in a room
            self.room_points.append(block_points), self.room_labels.append(block_labels)
        self.room_points = np.concatenate(self.room_points)
        self.room_labels = np.concatenate(self.room_labels)

        self.room_idxs = np.array(room_idxs)
        print("Totally {} samples in {} set.".format(len(self.room_idxs), split))

    def __getitem__(self, idx):  # get items in one block
        selected_points = self.room_points[idx]
        current_labels = self.room_labels[idx] 
        center = np.mean(selected_points, axis=0)
        N_points = selected_points.shape[0]

        # add normalized xyz
        current_points = np.zeros((N_points, 3))
        selected_points[:, 0] = selected_points[:, 0] - center[0]
        selected_points[:, 1] = selected_points[:, 1] - center[1]
        current_points[:, 0:3] = selected_points

        return current_points, current_labels, center

    def __len__(self):
        return len(self.room_idxs)


class S3DISDataset(Dataset):  # load data block by block, without using h5 files
    def __init__(self, split='train', data_root='trainval_fullarea', num_point=4096, block_size=1.0, sample_rate=1.0, num_class=20, use_all_points=False, num_thre = 1024):
        super().__init__()
        self.num_point = num_point
        self.block_size = block_size
        self.use_all_points = use_all_points
        self.num_thre = num_thre
        rooms = sorted(os.listdir(data_root))
        rooms = [room for room in rooms if 'Area_' in room]
        # test_areas = [1]
        print("NT: ", int(np.floor(len(rooms) * 1/8)))
        test_areas = np.random.choice(range(len(rooms)), int(np.floor(len(rooms) * 1/8)), replace = False)
        if split == "train":
            rooms_split = [room for room in rooms if not any(['Area_{}'.format(test_area) in room for test_area in test_areas])]
        else:
            rooms_split = [rooms[i] for i in test_areas]
        print("RS: ", len(rooms_split))
        print("TS: ", len(test_areas))
        self.room_points, self.room_labels = [], []
        self.room_coord_min, self.room_coord_max = [], []
        num_point_all = []
        labelweights = np.zeros(num_class)
        for room_name in rooms_split:
            room_path = os.path.join(data_root, room_name)
            room_data = np.load(room_path)  # xyzrgbl, N*7
            points, labels = room_data[:, 0:6], room_data[:, 6]  # xyzrgb, N*6; l, N
            tmp, _ = np.histogram(labels, range(num_class + 1))
            labelweights += tmp
            coord_min, coord_max = np.amin(points, axis=0)[:3], np.amax(points, axis=0)[:3]
            self.room_points.append(points), self.room_labels.append(labels)
            self.room_coord_min.append(coord_min), self.room_coord_max.append(coord_max)
            num_point_all.append(labels.size)
        labelweights = labelweights.astype(np.float32)
        labelweights = labelweights / np.sum(labelweights)
        self.labelweights = np.power(np.amax(labelweights) / labelweights, 1 / 3.0)
        print('label weights: ', self.labelweights)
        sample_prob = num_point_all / np.sum(num_point_all)
        # print("SP: ", sample_prob)
        num_iter = int(np.sum(num_point_all) * sample_rate / num_point)
        # print("NI: ", num_iter)
        room_idxs = []
        for index in range(len(rooms_split)):
            room_idxs.extend([index] * int(round(sample_prob[index] * num_iter)))  # extend with number of blocks in a room
        self.room_idxs = np.array(room_idxs)
        print("Totally {} samples in {} set.".format(len(self.room_idxs), split))

    def __getitem__(self, idx):  # get items in one block
        # print('here: ', threading.current_thread().ident)
        room_idx = self.room_idxs[idx]
        points = self.room_points[room_idx]   # N * 6

        labels = self.room_labels[room_idx]   # N

        N_points = points.shape[0]

        n_labs = range(int(np.max(np.unique(labels)) + 1))
        class_points = [[labels[i] for i in range(len(labels)) if labels[i] == j] for j in n_labs]
        class_lens = [len(class_points[i]) for i in n_labs]
        tot = sum(class_lens)
        shuf_nums = np.array([tot - i for i in class_lens])
        tot = sum(shuf_nums)
        weights = shuf_nums/tot
        weights.astype('float64')
        weights /= weights.sum()
        # tmp_weights = [class_lens[i]/tot for i in n_labs]
        # max_w = np.max(tmp_weights)

        # weights = [1 - weights[i] for i in n_labs]
        # weights = [weights[i]/sum(weights) for i in n_labs]

        point_weights = np.zeros(len(labels))
        # print(np.unique(labels))
        # print("Weights: ", weights)
        # print("SW: ", sum(weights))
        for i in range(len(labels)):
            point_weights[i] = weights[int(labels[i])]
        point_weights /= point_weights.sum()
        if np.isnan(sum(point_weights)):
            point_weights = [1/len(labels) for i in range(len(labels))]
        # print("PW: ", sum(point_weights))


        if self.use_all_points:
            self.num_point = N_points
            center = np.mean(points[:, :3], axis=0)
            selected_point_idxs = np.arange(N_points)

        else:
            while (True):
                center = points[np.random.choice(N_points, p = point_weights)][:3]
                block_min = center - [self.block_size / 2.0, self.block_size / 2.0, 0]
                block_max = center + [self.block_size / 2.0, self.block_size / 2.0, 0]
                point_idxs = np.where((points[:, 0] >= block_min[0]) & (points[:, 0] <= block_max[0]) & (points[:, 1] >= block_min[1]) & (points[:, 1] <= block_max[1]))[0]
                if point_idxs.size >= self.num_thre:
                    break
            if point_idxs.size >= self.num_point:
                this_pw = p = point_weights[point_idxs]
                this_pw /= this_pw.sum()
                selected_point_idxs = np.random.choice(point_idxs, self.num_point, p=this_pw, replace=False)
            else:
                this_pw = p = point_weights[point_idxs]
                this_pw /= this_pw.sum()
                selected_point_idxs = np.random.choice(point_idxs, self.num_point, p = this_pw, replace=True)

        # add normalized xyz
        center = points[np.random.choice(N_points)][:3]
        selected_points = points[selected_point_idxs, :]  # num_point * 6
        current_points = np.zeros((self.num_point, 9))  # num_point * 9
        current_points[:, 6] = selected_points[:, 0] / self.room_coord_max[room_idx][0]
        current_points[:, 7] = selected_points[:, 1] / self.room_coord_max[room_idx][1]
        current_points[:, 8] = selected_points[:, 2] / self.room_coord_max[room_idx][2]
        selected_points[:, 0] = selected_points[:, 0] - center[0]
        selected_points[:, 1] = selected_points[:, 1] - center[1]
        selected_points[:, 3:6] /= 255.0
        current_points[:, 0:6] = selected_points
        current_labels = labels[selected_point_idxs]

        return current_points, current_labels

    def __len__(self):
        return len(self.room_idxs)


class S3DISDataset_eval(Dataset):  # load data block by block, without using h5 files
    def __init__(self, split='train', data_root='trainval_fullarea', num_point=4096, test_area='5', block_size=1.0, stride=1.0, num_class=20, use_all_points=False, num_thre = 1024):
        super().__init__()
        self.num_point = num_point
        self.block_size = block_size
        
        self.use_all_points = use_all_points
        self.stride = stride
        self.num_thre = num_thre
        rooms = sorted(os.listdir(data_root))
        rooms = [room for room in rooms if 'Area_' in room]
        if split == 'train':
            rooms_split = [room for room in rooms if not 'Area_{}'.format(test_area) in room]
        else:
            if test_area == 'all':
                rooms_split = rooms
            else:
                rooms_split = [room for room in rooms if 'Area_{}'.format(test_area) in room]
        self.room_points, self.room_labels = [], []
        self.room_coord_min, self.room_coord_max = [], []

        room_idxs = []
        for index, room_name in enumerate(rooms_split):
            room_path = os.path.join(data_root, room_name)
            room_data = np.load(room_path)
            points, labels = room_data[:, 0:6], room_data[:, 6]
            # print("Total labels: ", len(labels))
            coord_min, coord_max = np.amin(points, axis=0)[:3], np.amax(points, axis=0)[:3]
            self.room_coord_min.append(coord_min), self.room_coord_max.append(coord_max)
            block_points, block_labels = util.room2blocks(points, labels, self.num_point, block_size=self.block_size,
                                                       stride=self.stride, random_sample=False, sample_num=None, use_all_points=self.use_all_points)
            # print("Block labels: ", block_labels.shape)
            room_idxs.extend([index] * int(block_points.shape[0]))  # extend with number of blocks in a room
            self.room_points.append(block_points), self.room_labels.append(block_labels)
        self.room_points = np.concatenate(self.room_points)
        self.room_labels = np.concatenate(self.room_labels)

        self.room_idxs = np.array(room_idxs)
        print("Rooms: ", self.room_idxs)
        print("Totally {} samples in {} set.".format(len(self.room_idxs), split))

    def __getitem__(self, idx):  # get items in one block
        room_idx = self.room_idxs[idx]
        selected_points = self.room_points[idx]   # num_point * 6
        current_labels = self.room_labels[idx]   # num_point
        center = np.mean(selected_points, axis=0)
        N_points = selected_points.shape[0]

        # add normalized xyz
        current_points = np.zeros((N_points, 9))  # num_point * 9
        current_points[:, 6] = selected_points[:, 0] / self.room_coord_max[room_idx][0]
        current_points[:, 7] = selected_points[:, 1] / self.room_coord_max[room_idx][1]
        current_points[:, 8] = selected_points[:, 2] / self.room_coord_max[room_idx][2]
        selected_points[:, 0] = selected_points[:, 0] - center[0]
        selected_points[:, 1] = selected_points[:, 1] - center[1]
        selected_points[:, 3:6] /= 255.0
        current_points[:, 0:6] = selected_points

        return current_points, current_labels

    def __len__(self):
        return len(self.room_idxs)