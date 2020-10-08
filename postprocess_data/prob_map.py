"""
Generate probability maps for a classified point cloud
Created by: Qian Bai
            on 23 August 2020
"""

import os
import numpy as np
from scipy.special import softmax
from tqdm import tqdm

DUMP_DIR = 'D:/Documents/Courses/Q5.0.Additional Thesis/visualization'
setting = 'NRI_30m_1m_k20'
test_area = '4'


def get_rooms_data(path):
    """
    Read two test results
    :param path1: folder path for test results
    :return: rooms_data. stacked room data (num_rooms,) with probability
    """
    num_rooms = len(os.listdir(path))
    room_idx = np.arange(0, num_rooms)

    rooms_data = []
    for room_id in tqdm(room_idx):
        filename = 'Area_%s_room_%d_pred_gt.txt' % (test_area, room_id)
        file = os.path.join(path, filename)

        # read test results
        result = np.loadtxt(file)
        # softmax logits
        result[:, 8:] = softmax(result[:, 8:], axis=1)
        rooms_data.append(result)

    return rooms_data


if __name__ == '__main__':
    test_path = os.path.join(DUMP_DIR, setting)

    print('Loading data ...')
    rooms_data = get_rooms_data(test_path)
    # save combined results
    print('Saving the file ...')
    num_rooms = len(rooms_data)
    OUTPUT_DIR = os.path.join(DUMP_DIR, 'Prob')
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    for i in range(num_rooms):
        filename = 'Area_%s_room_%d_pred_gt_prob.txt' % (test_area, i)
        np.savetxt(os.path.join(OUTPUT_DIR, filename), rooms_data[i], fmt='%f')