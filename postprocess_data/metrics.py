"""
Calculate overall accuracy, averaged class accuracy and mIOU
Created by: Qian Bai
        on 29 July 2020
"""

import os
import numpy as np
import sklearn.metrics as metrics
from tqdm import tqdm

DUMP_DIR = 'D:/Documents/Courses/Q5.0.Additional Thesis/visualization/Combined'
test_area = '4'


def get_rooms_result(path):
    """
    Read two test results
    :param path: folder path for test results
    :return: rooms_result, stacked room data (num_rooms,)
    """
    num_rooms = len(os.listdir(path))
    room_idx = np.arange(0, num_rooms)

    rooms_data = []
    for room_id in tqdm(room_idx):
        filename = 'Area_%s_room_%d_pred_gt_combined.txt' % (test_area, room_id)
        file1= os.path.join(path, filename)

        # read test results
        result = np.loadtxt(file1)
        # softmax logits
        rooms_data.append(result)

    return rooms_data


def calculate_sem_IoU(pred, gt, num_classes):
    I_all = np.zeros(num_classes)
    U_all = np.zeros(num_classes)
    for room_id in range(len(gt)):
        for cls in range(num_classes):
            I = np.sum(np.logical_and(pred[room_id] == cls, gt[room_id] == cls))
            U = np.sum(np.logical_or(pred[room_id] == cls, gt[room_id] == cls))
            I_all[cls] += I
            U_all[cls] += U
    return I_all / U_all


def calculate_accuracy(pred, gt):
    test_true_cls = np.concatenate(pred)
    test_pred_cls = np.concatenate(gt)
    test_acc = metrics.accuracy_score(test_true_cls, test_pred_cls)
    avg_per_class_acc = metrics.balanced_accuracy_score(test_true_cls, test_pred_cls)

    return test_acc, avg_per_class_acc


if __name__ == '__main__':
    print('Loading results ...')
    rooms_result = get_rooms_result(DUMP_DIR)
    num_rooms = len(rooms_result)
    pred, gt = [], []
    for i in range(num_rooms):
        pred.append(rooms_result[i][:, 6])
        gt.append(rooms_result[i][:, 7])

    mIoU = np.mean(calculate_sem_IoU(pred, gt, 4))
    acc, avg_per_class_acc = calculate_accuracy(pred, gt)
    print('Accuracy: %.6f, Avg accuracy: %.6f, mean IoU: %.6f' % (acc, avg_per_class_acc, mIoU))
