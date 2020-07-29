"""
This file is to combine two test results with different effective ranges (block sizes ...)
Created by: Qian Bai
            on 29 July 2020
"""

import os
import numpy as np
from scipy.special import softmax
from tqdm import tqdm

DUMP_DIR = 'D:/Documents/Courses/Q5.0.Additional Thesis/visualization'
setting1 = 'NRI_30m_1m_k20'
setting2 = 'NRI_50m_1m_k20'
test_area = '4'


def get_rooms_data(path1, path2):
    """
    Read two test results
    :param path1: folder path for test results 1
    :param path2: folder path for test results 2
    :return: rooms_data1, rooms_data2. stacked room data (num_rooms,)
    """
    num_rooms = len(os.listdir(path1))
    room_idx = np.arange(0, num_rooms)

    rooms_data1, rooms_data2 = [], []
    for room_id in tqdm(room_idx):
        filename = 'Area_%s_room_%d_pred_gt.txt' % (test_area, room_id)
        file1 = os.path.join(path1, filename)
        file2 = os.path.join(path2, filename)

        # read 2 test results
        result1 = np.loadtxt(file1)
        result2 = np.loadtxt(file2)
        # softmax logits
        result1[:, 8:] = softmax(result1[:, 8:], axis=1)
        result2[:, 8:] = softmax(result2[:, 8:], axis=1)
        rooms_data1.append(result1)
        rooms_data2.append(result2)

    return rooms_data1, rooms_data2


def combine(data1, data2):
    """
    Compare and combine two test results room by room
    :param data1:
    :param data2:
    :return: rooms_result
    """
    num_rooms = len(data1)
    room_idx = np.arange(0, num_rooms)

    rooms_result = []
    for room_id in tqdm(room_idx):
        room_data1 = data1[room_id]
        room_data2 = data2[room_id]
        # find co-locations
        xyz1, xyz2 = room_data1[:, :3], room_data2[:, :3]
        xyz1 = [tuple(x) for x in xyz1]
        xyz2 = [tuple(x) for x in xyz2]
        ind1_dict = dict((k, i) for i, k in enumerate(xyz1))
        ind2_dict = dict((k, i) for i, k in enumerate(xyz2))
        colocations = set(xyz1).intersection(set(xyz2))
        final_idx1 = [ind1_dict[x] for x in colocations]
        final_idx2 = [ind2_dict[x] for x in colocations]

        # compare results at co-locations
        room_result = []
        for i, id in enumerate(final_idx1):
            xyz = room_data1[id, :3]
            rni = room_data1[id, 3:6]
            pred1 = room_data1[id, 6]
            pred2 = room_data2[final_idx2[i], 6]
            gt = room_data1[id, 7]
            prob1 = room_data1[id, 8:]
            prob2 = room_data2[final_idx2[i], 8:]

            if pred1 == pred2:
                pred = pred1
            else:
                pred = pred1 if prob1[int(pred1)] >= prob2[int(pred2)] else pred2
            room_result.append(np.append(np.append(xyz, rni), [pred, gt]))

        room_result = np.array(room_result)
        rooms_result.append(room_result)

    return rooms_result


if __name__ == '__main__':
    test_path1 = os.path.join(DUMP_DIR, setting1)
    test_path2 = os.path.join(DUMP_DIR, setting2)

    print('Loading data ...')
    rooms_data1, rooms_data2 = get_rooms_data(test_path1, test_path2)
    print('Combining two results ...')
    combined_result = combine(rooms_data1, rooms_data2)
    # save combined results
    print('Saving the combined file ...')
    num_rooms = len(combined_result)
    OUTPUT_DIR = os.path.join(DUMP_DIR, 'Combined')
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    for i in range(num_rooms):
        filename = 'Area_%s_room_%d_pred_gt_combined.txt' % (test_area, i)
        np.savetxt(os.path.join(OUTPUT_DIR, filename), combined_result[i], fmt='%f')