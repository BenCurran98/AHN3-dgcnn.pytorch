"""
Write paths of annotation folders in a text file
Created by: Qian Bai
            on 9 July 2020
"""
import os
import glob
import numpy as np

BASE_DIR = 'D:/Documents/Datasets/'  # base directory of datasets
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, 'AHN3_as_S3DIS_RGB')

with open(os.path.join(ROOT_DIR, 'meta/anno_paths.txt'), 'w+') as anno_paths:
    paths = []
    for path in glob.glob(DATA_PATH + '/*/*/Annotations'):
        path = path.replace('\\', '/')
        paths.append(path)
    paths = np.array(paths)
    np.savetxt(anno_paths, paths, fmt='%s')
anno_paths.close()
