"""
Write file names of data_label.npy in a text file
Created by: Qian Bai
            on 9 July 2020
"""
import os
import glob
import numpy as np
import argparse

BASE_DIR = os.path.join(os.getcwd(), '..', 'Datasets')
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, 'powercor_as_S3DIS_NRI_NPY')

def write_npy_file_names(base_dir, root_dir, data_path):
    with open(os.path.join(root_dir, 'meta/all_data_label.txt'), 'w+') as f:
        files = []
        for file in glob.iglob(data_path + '/*.npy'):
            print(file)
            file = os.path.basename(file)
            files.append(file)
        paths = np.array(files)
        np.savetxt(f, files, fmt='%s')
    f.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Extract point cloud data')
    parser.add_argument('--base_dir', type = str, default = BASE_DIR, help = 'Base directory of data')
    parser.add_argument('--root_dir', type = str, default = ROOT_DIR, help = 'Root directory of data')
    parser.add_argument('--data_path', type = str, default = DATA_PATH, help = 'Folder containing pointcloud text data')
    
    args = parser.parse_args()
    write_npy_file_names(args.base_dir, args.root_dir, args.data_path)
    