"""
Write file names of data_label.npy in a text file
Created by: Qian Bai
            on 9 July 2020
"""
import os
import glob
import numpy as np

BASE_DIR = 'D:/Documents/Datasets/'  # base directory of datasets
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, 'AHN3_as_S3DIS_RGB_NPY')

with open(os.path.join(ROOT_DIR, 'meta/all_data_label.txt'), 'w+') as f:
    files = []
    for file in glob.iglob(os.path.join(DATA_PATH, '*.npy')):
        file = os.path.basename(file)
        files.append(file)
    paths = np.array(files)
    np.savetxt(f, files, fmt='%s')
f.close()