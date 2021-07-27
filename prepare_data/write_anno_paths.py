"""
Write paths of annotation folders in a text file
Created by: Qian Bai
            on 9 July 2020
"""
import os
import glob
import numpy as np
import argparse

BASE_DIR = os.path.join(os.getcwd(), '..', 'Datasets')
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, 'powercor_as_S3DIS_NRI')

def write_anno_paths(base_dir, root_dir, data_path):
    if not os.path.isdir(data_path):
        os.mkdir(data_path)

    with open(os.path.join(root_dir, 'meta/anno_paths.txt'), 'w+') as anno_paths:
        paths = []
        for path in glob.glob(base_dir + '/powercor_processed/*/*/Annotations'):
            path = path.replace('\\', '/')
            paths.append(path)
        paths = np.array(paths)
        np.savetxt(anno_paths, paths, fmt='%s')
    anno_paths.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Extract point cloud data')
    parser.add_argument('--base_dir', type = str, default = BASE_DIR, help = 'Base directory of data')
    parser.add_argument('--root_dir', type = str, default = ROOT_DIR, help = 'Root directory of data')
    parser.add_argument('-data_path', type = str, default = DATA_PATH, help = 'Folder containing pointcloud text data')
    
    args = parser.parse_args()
    write_anno_paths(args.base_dir, args.root_dir, args.data_path)
    