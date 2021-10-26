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
    validation_prop : float
        Fraction of data set to use as test/validation data (default 0.2)
    classes : list
        List of classes used in the data set (default [1, 2, 3, 4, 5])
    sample_num : int
        Number of blocks to randomly sample from each tile (default 5)
    """
    def __init__(self, split='train', data_root='', num_point=4096, 
                    use_all_points=False, validation_prop = 0.2, 
                    classes = [0, 1, 2, 3, 4]):
        super().__init__()
        self.num_point = num_point
        self.use_all_points = use_all_points
        self.validation_prop = validation_prop
        self.classes = classes
        rooms = sorted(os.listdir(data_root))
        rooms = [room for room in rooms if 'Area_' in room]

        test_areas = np.random.choice(range(len(rooms)), int(np.floor(len(rooms) * self.validation_prop)), replace = False)
        if split == "train":
            rooms_split = [room for room in rooms if not any(['Area_{}'.format(test_area) in room for test_area in test_areas])]
        else:
            rooms_split = [rooms[i] for i in test_areas]
            
        self.room_points, self.room_labels = [], []
        self.room_coord_min, self.room_coord_max = [], []

        for index in tqdm(range(len(rooms_split)), "Loading Tiles"):
            room_name = rooms_split[index]
            room_path = os.path.join(data_root, room_name)
            room_data = np.load(room_path)
            points, labels = room_data[:, 0:-1], room_data[:, -1]
            coord_min, coord_max = np.amin(points, axis=0)[:3], np.amax(points, axis=0)[:3]
            self.room_coord_min.append(coord_min), self.room_coord_max.append(coord_max)

            self.room_points.append(np.reshape(points, (1, points.shape[0], points.shape[1])))
            self.room_labels.append(np.reshape(labels, (1, labels.shape[0])))

        self.room_points = np.concatenate(self.room_points)
        self.room_labels = np.concatenate(self.room_labels)

        self.room_idxs = np.array([i for i in range(len(rooms_split))])

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

        N_points = points.shape[0]

        if not self.use_all_points:
            N_points = self.num_point

        # subsample the points
        selected_point_idxs = self.sample_points(idx, N_points)

        selected_points = points[selected_point_idxs, :]
        selected_labels = labels[selected_point_idxs]

        return selected_points, selected_labels, idx

    def __len__(self):
        return len(self.room_idxs)


def collate_pcs(data):
    points, labels, idxs = zip(*data)
    num_points = [points[i].shape[0] for i in range(len(points))]
    min_num_points = min(num_points)
    batch_points = np.zeros((len(points), min_num_points, points[0].shape[1]))
    batch_labels = np.zeros((len(points), min_num_points))
    for i in range(len(points)):
        subsampled_idxs = np.random.choice(range(points[i].shape[0]), 
                                            min_num_points, 
                                            replace = False)
        batch_points[i, :, :] = points[i][subsampled_idxs, :]
        batch_labels[i, :] = labels[i][subsampled_idxs]

    return torch.tensor(batch_points), torch.tensor(batch_labels), torch.tensor(idxs)