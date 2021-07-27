import os
import indoor3d_util as utils
from tqdm import tqdm
import argparse

BASE_DIR = os.path.join(os.getcwd(), '..', 'Datasets')
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, 'powercor_processed')

def collect_indoor_3d_data(root_dir, output_folder):
    anno_paths = [line.rstrip() for line in open(os.path.join(root_dir, 'meta/anno_paths.txt'))]

    if not os.path.exists(output_folder):
        os.mkdir(output_folder)

    for anno_path in tqdm(anno_paths):
        elements = anno_path.split('/')
        out_filename = elements[-3] + '_' + elements[-2] + '.npy'  # room name: Area_1_38FN1_1.npy
        utils.collect_point_label(anno_path, os.path.join(output_folder, out_filename), 'numpy')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Collect processed data')
    parser.add_argument('--root_dir', type = str, default = ROOT_DIR, help = 'Root directory of data')
    parser.add_argument('--output_folder', type = str, default = os.path.join(BASE_DIR, 'powercor_as_S3DIS_NRI_NPY'), help = 'Output folder of the data summary')

    args = parser.parse_args()

    collect_indoor_3d_data(args.root_dir, args.output_folder)