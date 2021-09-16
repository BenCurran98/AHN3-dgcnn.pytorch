import os
import numpy as np
from scipy.special import softmax
from tqdm import tqdm
import laspy
import argparse
from tqdm import tqdm

from prepare_data.pointcloud_util import g_label2color

def get_predictions(pred_file, las_file):
    result = np.loadtxt(pred_file)
    # probs = softmax(result[:, 8:], axis = 1)
    # labels = np.argmax(probs, axis = 1) 
    print(result.shape)
    labels = result[:, 3]
    points = result[:, 0:3]
    colors = [g_label2color[c] for c in labels]

    las = laspy.create(file_version = "1.2", point_format = 3) 

    las.x = points[:, 0]
    las.y = points[:, 1]
    las.z = points[:, 2]

    las.red = [c[0] for c in colors]
    las.green = [c[1] for c in colors]
    las.blue = [c[2] for c in colors]


    las.classification = labels

    las.write(las_file)

    return points, labels

def get_predictions_dir(pred_dir, out_dir):
    all_files = [f for f in os.listdir(pred_dir) if os.path.isfile(os.path.join(pred_dir, f))]
    pred_files = [f for f in all_files if f[-11:-4] == "pred_gt"]

    pred_files = sorted(pred_files, key = str.lower)
    
    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)

    for i in tqdm(range(len(pred_files)), "Reading Pointcloud Predictions"):
        out_las_name = "{}.las".format(pred_files[i][:-4])
        out_file = os.path.join(out_dir, out_las_name)

        get_predictions(os.path.join(pred_dir, pred_files[i]), out_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Extract DGCNN pointcloud predicitons')
    parser.add_argument('--pred_dir', type = str, default = "predict", help = 'Directory of DGCNN predictions')
    parser.add_argument('--out_dir', type = str, default = "predict_las", help = 'Directory to save LAS prediction files to')
    
    args = parser.parse_args()

    get_predictions_dir(args.pred_dir, args.out_dir)