"""
Extract features in AHN3 point clouds and save in S3DIS format
Created by: Qian Bai
            on 8 July 2020
"""
import os
import glob
import numpy as np
import json
import h5py
from tqdm import tqdm
import argparse

AREA_ID = 1
AREA = 'some_tile'  # 1: 38FN1, 2: 37EN2, 3: 32CN1, 4: 31HZ2

# BASE_DIR = '/home/ubuntu/Datasets/'
BASE_DIR = os.path.join(os.getcwd(), '..' '/Datasets')
#   # base directory of datasets
DATA_FOLDER = os.path.join(BASE_DIR, 'powercor', AREA)  # path of subsampled AHN3 point clouds


def extract_annotations(area_id, area, data_folder, output_path, categories, features, features_output):
    if not os.path.exists(output_path):
        os.mkdir(output_path)

    print('Generate files for Area {}'.format(area_id))

    print("data_folder = ", data_folder)

    orig_output_path = output_path
    if not os.path.exists(output_path):
        os.mkdir(output_path)

    for room_id, room_file in tqdm(enumerate(glob.iglob(os.path.join(data_folder, '*.txt')))):  # read each tiled point cloud
        print("room = ", room_file)
        room_id += 1
        output_path = orig_output_path
        output_path = os.path.join(output_path, 'Area_' + str(room_id))
        if not os.path.exists(output_path):
            os.mkdir(output_path)

        output_path = os.path.join(output_path, area)
        if not os.path.exists(output_path):
            os.mkdir(output_path)

        # load data
        room_data = np.loadtxt(room_file)
        output_label = room_data[:, -1]
        # merge bridges into ground
        bridge_idx = np.where(output_label == 26.0)[0]
        output_label[bridge_idx] = 2.0
        output_data = np.zeros((room_data.shape[0], len(features_output)))
        test = np.unique(output_label)
        # select the output features
        for feature_id, feature in enumerate(features_output):
            output_data[:, feature_id] = room_data[:, features[feature]]
        with open(output_path + '/' + area + '_' + str(room_id) + '.txt', 'w+') as fout1:
            np.savetxt(fout1, output_data, fmt="%.3f %.3f %.3f %d %d %d")
        fout1.close()

        # write file according to classes
        ANNO_PATH = os.path.join(output_path, 'Annotations')
        if not os.path.exists(ANNO_PATH):
            os.mkdir(ANNO_PATH)

        eff_categories = np.unique(output_label)
        for category in eff_categories:
            # find corresponding classes
            category_indices = np.where(output_label == category)[0]
            with open(ANNO_PATH + '/' + categories[category] + '.txt', 'w+') as fout2:
                np.savetxt(fout2, output_data[category_indices, :], fmt="%.3f %.3f %.3f %d %d %d")
            fout2.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Extract point cloud data')
    parser.add_argument('--base_dir', type = str, default = BASE_DIR, help = 'Base directory of data')
    parser.add_argument('--area_id', type = int, default = AREA_ID, help = 'ID of the area to extract')
    parser.add_argument('--area', type = str, default = AREA, help = 'Name of area to process')
    parser.add_argument('--data_folder', type = str, default = DATA_FOLDER, help = 'Folder containing the complete datasets')
    parser.add_argument('--categories_file', type = str, default = 'params/categories.json', help = 'JSON file containing label mappings')
    parser.add_argument('--features_file', type = str, default = 'params/features.json', help = 'JSON file containing index mappings of LiDAR features')
    parser.add_argument('--features_output', nargs = '*', type = str, default = ['X', 'Y', 'Z', 'NumberOfReturns', 'ReturnNumber', 'Intensity'], help = 'LiDAR features to extract')
    args = parser.parse_args()
    output_folder = os.path.join(args.base_dir, 'powercor_processed')
    f = open(args.categories_file, 'r')
    categories = json.load(f)
    f.close()
    orig_categories = categories
    categories = {float(k): v for k, v in orig_categories.items()}

    f = open(args.features_file, 'r')
    features = json.load(f)
    f.close()
    extract_annotations(args.area_id, args.area, args.data_folder, output_folder, categories, features, args.features_output)
