import os
import prepare_data.indoor3d_util as utils
from tqdm import tqdm

BASE_DIR = 'D:/Documents/Datasets/'  # base directory of datasets
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, 'AHN3_as_S3DIS_NRI')

anno_paths = [line.rstrip() for line in open(os.path.join(ROOT_DIR, 'meta/anno_paths.txt'))]
# anno_paths = [os.path.join(DATA_PATH, p) for p in anno_paths]

output_folder = os.path.join(BASE_DIR, 'AHN3_as_S3DIS_NRI_NPY')
if not os.path.exists(output_folder):
    os.mkdir(output_folder)

""" revise mistake in S3DIS dataset """
""" 
revise_file = os.path.join(DATA_PATH, "Area_5/hallway_6/Annotations/ceiling_1.txt")
with open(revise_file, "r") as f:
    data = f.read()
    data = data[:5545347] + ' ' + data[5545348:]
    f.close()
with open(revise_file, "w") as f:
    f.write(data)
    f.close()
"""

for anno_path in tqdm(anno_paths):
    print(anno_path)
    elements = anno_path.split('/')
    out_filename = elements[-3] + '_' + elements[-2] + '.npy'  # room name: Area_1_38FN1_1.npy
    utils.collect_point_label(anno_path, os.path.join(output_folder, out_filename), 'numpy')
