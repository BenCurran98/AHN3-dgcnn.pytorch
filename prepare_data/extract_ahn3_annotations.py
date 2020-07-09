"""
Extract features in AHN3 point clouds and save in S3DIS format
Created by: Qian Bai
            on 8 July 2020
"""
import os
import glob
import numpy as np
from tqdm import tqdm

# catogories in AHN3 dataset
categories = {2.0:  'ground',
              1.0:  'vegetation',
              6.0:  'building',
              26.0: 'bridge',
              9.0: 'water'}

# features and id in AHN3 dataset
AHN3_features = {'X': 0,
                 'Y': 1,
                 'Z': 2,
                 'R': 3,
                 'G': 4,
                 'B': 5,
                 'PointSourceId': 6,
                 'UserData': 7,
                 'ScanAngleRank': 8,
                 'EdgeOfFlightLine': 9,
                 'NumberOfReturns': 10,
                 'ReturnNumber': 11,
                 'GpsTime': 12,
                 'Intensity': 13,
                 'Class': 14}
features_output = ['X', 'Y', 'Z', 'R', 'G', 'B']

AREA_ID = 4
AREA = '31HZ2'  # 1: 38FN1, 2: 37EN2, 3: 32CN1, 4: 31HZ2

BASE_DIR = 'D:/Documents/Datasets/'  # base directory of datasets
DATA_FOLDER = os.path.join(BASE_DIR, 'AHN3_subsampled_1m', AREA)  # path of subsampled AHN3 point clouds
OUTPUT_FOLDER = os.path.join(BASE_DIR, 'AHN3_as_S3DIS_RGB')

print('Generate files for Area {}'.format(AREA_ID))

for room_id, room_file in tqdm(enumerate(glob.iglob(os.path.join(DATA_FOLDER, '*.txt')))):  # read each tiled point cloud
    room_id += 1
    OUTPUT_PATH = os.path.join(OUTPUT_FOLDER, 'Area_' + str(AREA_ID), AREA + '_' + str(room_id))
    if not os.path.exists(OUTPUT_PATH):
        os.mkdir(OUTPUT_PATH)

    # load data
    room_data = np.loadtxt(room_file)
    output_label = room_data[:, -1]
    output_data = np.zeros((room_data.shape[0], len(features_output)))
    test = np.unique(output_label)
    # select the output features
    for feature_id, feature in enumerate(features_output):
        output_data[:, feature_id] = room_data[:, AHN3_features[feature]]
    with open(OUTPUT_PATH + '\\' + AREA + '_' + str(room_id) + '.txt', 'w+') as fout1:
        np.savetxt(fout1, output_data, fmt="%.3f %.3f %.3f %d %d %d")
    fout1.close()

    # write file according to classes
    ANNO_PATH = os.path.join(OUTPUT_PATH, 'Annotations')
    if not os.path.exists(ANNO_PATH):
        os.mkdir(ANNO_PATH)

    eff_categories = np.unique(output_label)
    for category in eff_categories:
        # find corresponding classes
        category_indices = np.where(output_label == category)[0]
        with open(ANNO_PATH + '\\' + categories[category] + '.txt', 'w+') as fout2:
            np.savetxt(fout2, output_data[category_indices, :], fmt="%.3f %.3f %.3f %d %d %d")
        fout2.close()
