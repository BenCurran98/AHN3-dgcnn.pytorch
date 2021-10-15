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
    cell_size : float
        cell_size (num_point/metre^3) to sample points at (default 0.4641588833612779)
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
    def __init__(self, split='train', data_root='',
                    cell_size = 0.4641588833612779,
                    block_size=30.0, use_all_points=False, test_prop = 0.2, 
                    classes = [0, 1, 2, 3, 4], sample_num = 5, class_min = 100,
                    n_tries = 10, fields = []):
        super().__init__()
        self.cell_size = cell_size
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
        with tqdm(range(len(rooms_split)), "Sampling Tiles") as t:
            for index in range(len(rooms_split)):
                room_name = rooms_split[index]
                room_path = os.path.join(data_root, room_name)
                room_data = np.load(room_path)
                points, labels = room_data[:, 0:-1], room_data[:, -1]
                coord_min, coord_max = np.amin(points, axis=0)[:3], np.amax(points, axis=0)[:3]
                self.room_coord_min.append(coord_min), self.room_coord_max.append(coord_max)

                las = laspy.create(file_version = "1.2", point_format = 3) 

                las.x = points[:, 0]
                las.y = points[:, 1]
                las.z = points[:, 2]

                if "color" in fields:
                    las.red = points[:, 3]
                    las.green = points[:, 4]
                    las.blue = points[:, 5]
                if "intensity" in fields:
                    las.intensity = points[:, 6]
                if "return_number" in fields:
                    las.return_number = points[:, 7]
                if "number_of_returns" in fields:
                    las.number_of_returns = points[:, 8]

                las.classification = labels
                
                unique_labels = np.unique(labels)
                
                label_counts = [len(np.where(labels == c)[0]) for c in unique_labels]

                las.write("../DataSampleTrain/{}_block_data{}.las".format(split, index))

                if split == "train": 
                    found = 0
                    n = 0
                    while found < sample_num:
                        block_points, block_labels = util.room2blocks(points, labels, cell_size = self.cell_size, block_size=self.block_size,
                                                                stride=self.block_size/10, random_sample=True, sample_num=sample_num - found, use_all_points=self.use_all_points)
                        for i in range(len(block_points)):
                            this_block_points = block_points[i]
                            this_block_labels = block_labels[i]
                            label_counts = [len(np.where(this_block_labels == c)[0]) for c in classes]
                            if all([c > class_min for c in label_counts]):
                                found += 1
                                room_idxs.extend([index])
                                self.room_points.append(np.reshape(this_block_points, (1, this_block_points.shape[0], this_block_points.shape[1])))
                                self.room_labels.append(np.reshape(this_block_labels, (1, this_block_labels.shape[0])))
                                num_point_all.append(this_block_labels.size)
                        n += 1

                        if n > n_tries:
                            break
                    
                    if len(self.room_points) > 0:
                        las = laspy.create(file_version = "1.2", point_format = 3) 

                        las.x = self.room_points[-1][0, :, 0]
                        las.y = self.room_points[-1][0, :, 1]
                        las.z = self.room_points[-1][0, :, 2]

                        if "color" in fields:
                            las.red = points[:, 3]
                            las.green = points[:, 4]
                            las.blue = points[:, 5]
                        if "intensity" in fields:
                            las.intensity = points[:, 6]
                        if "return_number" in fields:
                            las.return_number = points[:, 7]
                        if "number_of_returns" in fields:
                            las.number_of_returns = points[:, 8]

                        las.classification = self.room_labels[-1][0, :]

                        las.write("../DataSampleTrain/{}_subsampled__block_data{}.las".format(split, index))

                else:
                    block_points, block_labels = util.room2blocks(points, labels, cell_size = self.cell_size, block_size=self.block_size,
                                                                stride=self.block_size/3, random_sample=True, sample_num=sample_num, use_all_points=self.use_all_points)
                    room_idxs.extend([index] * int(len(block_points)))  # extend with number of blocks in a room
                    self.room_points.append(block_points), self.room_labels.append(block_labels)
                    num_point_all.append(labels.size)

                t.set_postfix(num_samples = len(room_idxs))
                t.update()
        # self.room_points = np.concatenate(self.room_points)
        # self.room_labels = np.concatenate(self.room_labels)

        self.room_idxs = np.array(room_idxs)

        print("Totally {} samples in {} set.".format(len(self.room_idxs), split))

    def create_train_mask(self, idx, tot_samples, exclude_classes = []):
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
        min_label_count = min([label_counts[i] for i in range(len(label_counts)) if i not in exclude_classes])
        n_samples = int(min(min_label_count, np.floor(tot_samples/len(self.classes))))
        train_mask = np.zeros(labels.shape)
        for label in self.classes:
            this_label_idxs = np.where(labels == label)[0]
            if len(this_label_idxs) > 0:
                if label in exclude_classes:
                    continue
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
        
        return points, labels, idx

    def __len__(self):
        return len(self.room_idxs)

class FugroDataset_eval(Dataset):
    def __init__(self, split='train', data_root='', cell_size = 0.4641588833612779, block_size=30.0, use_all_points=False):
        super().__init__()
        self.block_size = block_size
        self.cell_size = cell_size
        
        self.use_all_points = use_all_points
        rooms = sorted(os.listdir(data_root))
        rooms_split = [room for room in rooms if 'Area_' in room]
        
        self.room_points, self.room_labels = [], []
        self.room_coord_min, self.room_coord_max = [], []
        self.centres = []

        room_idxs = []

        print(rooms_split)
        print(len(rooms_split))
        for index in tqdm(range(len(rooms_split)), "Sampling Tiles"):
        # for index in tqdm(range(1), "Samplng Tiles"):
            room_name = rooms_split[index]
            room_path = os.path.join(data_root, room_name)
            room_path = os.path.join(data_root, room_name)
            room_data = np.load(room_path)
            points, labels = room_data[:, 0:-1], room_data[:, -1]
            coord_min, coord_max = np.amin(points, axis=0)[:3], np.amax(points, axis=0)[:3]
            self.room_coord_min.append(coord_min), self.room_coord_max.append(coord_max)
            block_points, block_labels = util.room2blocks(points, labels, cell_size = self.cell_size, block_size=self.block_size,
                                                       stride=self.block_size, random_sample=False, sample_num=None, use_all_points=self.use_all_points)
            f = open("../DataSampleTest/block_data{}.txt".format(index), "w")
            these_points = np.concatenate(block_points, 1)[0, :]
            for i in range(these_points.shape[0]):
                f.write("%f %f %f\n" % (these_points[i, 0], these_points[i, 1], these_points[i, 2]))
            f.close()
            room_idxs.extend([index] * int(len(block_points)))  # extend with number of blocks in a room
            self.room_points.append(block_points), self.room_labels.append(block_labels)

        self.room_idxs = np.array(room_idxs)
        print("Totally {} samples in {} set.".format(len(self.room_idxs), split))

    def __getitem__(self, idx):  # get items in one block
        selected_points = self.room_points[idx]
        current_labels = self.room_labels[idx] 

        return selected_points, current_labels

    def __len__(self):
        return len(self.room_idxs)

def pc_collate(data):
    _, labels, room_idxs = zip(*data)
    lengths = [len(labs) for labs in labels]
    min_num_points = min(lengths)
    features = torch.zeros(len(data), min_num_points, data[0][0].shape[1]) # (batch_size, num_points, num_features)
    batch_labels = torch.zeros(len(data), len(labels))
    for i in range(len(data)):
        num_points = data[i][0].shape[0]
        selected_idxs = np.random.choice(range(num_points), min_num_points, replace = False)
        features[i, :, :] = data[i][0][selected_idxs, :]
        batch_labels[i, :] = data[i][1][selected_idxs]
    
    return features, batch_labels, room_idxs