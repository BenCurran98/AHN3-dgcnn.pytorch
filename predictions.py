import os
import numpy as np
from scipy.special import softmax
from tqdm import tqdm
import laspy

from prepare_data.pointcloud_util import g_label2color

def get_predictions(pred_file, las_file):
    result = np.loadtxt(pred_file)
    probs = softmax(result[:, 8:], axis = 1)
    labels = np.argmax(probs, axis = 1) 
    points = result[:, 0:3]
    colors = [g_label2color[c] for c in labels]

    las = laspy.create(file_version = "1.2", point_format = 3) 

    las.x = points[:, 0]
    las.y = points[:, 1]
    las.z = points[:, 2]

    print(colors)

    las.red = [c[0] for c in colors]
    las.green = [c[1] for c in colors]
    las.blue = [c[2] for c in colors]

    print("HI")

    las.classification = labels

    las.write(las_file)

    return points, labels