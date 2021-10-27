import numpy as np
import glob
import os
import sys
from tqdm import tqdm
import json

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(ROOT_DIR)
CLASS_NAMES_FILE = os.path.join(ROOT_DIR, 'meta/class_names4.txt')


def collect_point_label(anno_path, out_filename, file_format='txt', class_names_file = CLASS_NAMES_FILE):
    """ Convert original dataset files to data_label file (each line is XYZRGBL).
        We aggregated all the points from each instance in the room.

    Args:
        anno_path (str): path to annotations
        out_filename (str): path to save collected points and labels
        file_format (str): txt or numpy, determines what file format to save. Defaults to "txt"

    Returns:
        None
    Note:
        the points are shifted before save, the most negative point is now at origin.
    """

    g_classes = [x.rstrip() for x in open(class_names_file)]
    g_class2label = {cls: i for i, cls in enumerate(g_classes)}

    points_list = []

    for f in glob.glob(os.path.join(anno_path, '*.txt')):
        cls = os.path.basename(f).split('.')[0] # TODO: '_', changed because points in the same class are not separated
        if cls not in g_classes:  # note: in some room there is 'staris' class..
            cls = 'noise'
        
        points = np.loadtxt(f)
        if len(points.shape) == 1:
            points = np.reshape(points, (1, points.shape[0]))
        labels = np.ones((points.shape[0], 1)) * g_class2label[cls]
        points_list.append(np.concatenate([points, labels], 1))  # Nx7

    data_label = np.concatenate(points_list, 0)
    xyz_min = np.amin(data_label, axis=0)[0:3]
    data_label[:, 0:3] -= xyz_min

    if file_format == 'txt':
        fout = open(out_filename, 'w+')
        for i in range(data_label.shape[0]):
            str_format = " ".join(["%f" for _ in range(data_label.shape[1] - 1)])
            str_format += " %d\n"
            fout.write(str_format % tuple([j for j in data_label[i, :]]))
        fout.close()
    elif file_format == 'numpy':
        np.save(out_filename, data_label)
    else:
        print('ERROR!! Unknown file format: %s, please use txt or numpy.' % \
              (file_format))
        exit()

def sample_data(data, num_sample):
    """Randomly sample from data

    Args:
        data (array-like): NxF matrix of point cloud features
        num_sample (Int): Number of points to subsample. If num_sample > N, return original `data`
    """
    N = data.shape[0]
    if (N == num_sample):
        return data, range(N)
    elif (N > num_sample):
        sample = np.random.choice(N, num_sample)
        return data[sample, ...], sample
    else:
        sample = np.random.choice(N, num_sample - N)
        dup_data = data[sample, ...]
        return np.concatenate([data, dup_data], 0), list(range(N)) + list(sample)

def sample_data_label(data, label, num_sample):
    """Subsample point cloud data and labels

    Args:
        data (array-like): NxF matrix of point cloud features
        label (array-like): Nx1 array of classification labels
        num_sample (Int): Number of points to subsample. If num_sample > N, return original `data`

    Returns:
        new_data (ndarray): Array of subsampled data
        new_label (ndarray): Array of subsampled labels
    """
    new_data, sample_indices = sample_data(data, num_sample)
    new_label = label[sample_indices]
    return new_data, new_label


def room2blocks(data, label, num_point, 
                block_size=100.0, 
                stride=50.0,
                random_sample=False, 
                sample_num=None,
                use_all_points=False):
    """ Prepare block training data.
    Args:
        data (array-like): N x F numpy array, where N is the number of points, F is the number of point features
        label (array-like): N size uint8 numpy array of classification labels
        num_point (int): how many points to sample in each block
        block_size (float): physical size of the block in meters. Defaults to 50
        stride (float): stride for block sweeping. Defaults to 100
        random_sample (bool): if True, we will randomly sample blocks in the room. Defaults to False
        sample_num (int): if random sample, how many blocks to sample. Defaults to None
        use_all_points (bool): whether to use all points present in each tile. Defaults to False
    Returns:
        block_data_return (ndarray): B x num_point x F array of batched point cloud tiled data (B batches)
        block_labels (ndarray): B x num_point x 1 array of uint8 tile classification labels
        
    TODO: for this version, blocking is in fixed, non-overlapping pattern.
    """
    assert (stride <= block_size)

    x_ub = np.amax(data[:, 0])
    x_lb = np.amin(data[:, 0])
    
    y_ub = np.amax(data[:, 1])
    y_lb = np.amin(data[:, 1])

    # Get the corner location for our sampling blocks    
    xbeg_list = []
    ybeg_list = []
    if not random_sample:
        num_block_x = int(np.ceil(((x_ub - x_lb) - block_size) / stride)) + 1
        num_block_y = int(np.ceil(((y_ub - y_lb) - block_size) / stride)) + 1
        for i in range(num_block_x):
            for j in range(num_block_y):
                xbeg_list.append(x_lb + i * stride)
                ybeg_list.append(y_lb + j * stride)
    else:
        num_block_x = int(np.ceil((x_ub - x_lb) / block_size))
        num_block_y = int(np.ceil((y_ub - y_lb) / block_size))
        if sample_num is None:
            sample_num = num_block_x * num_block_y
        for _ in range(sample_num):
            xbeg = np.random.uniform(x_lb, x_ub)
            ybeg = np.random.uniform(y_lb, y_ub)
            xbeg_list.append(xbeg)
            ybeg_list.append(ybeg)
            
    # Collect blocks
    block_data_list = []
    block_label_list = []
    idx = 0
    for idx in range(len(xbeg_list)):
        xbeg = xbeg_list[idx]
        ybeg = ybeg_list[idx]

        if random_sample:
            found = False
            while not found:
                xcond = (data[:, 0] <= xbeg + block_size) & (data[:, 0] >= xbeg)
                ycond = (data[:, 1] <= ybeg + block_size) & (data[:, 1] >= ybeg)
                cond = xcond & ycond
                if np.sum(cond) < 1000:  # discard block if there are less than 100 pts.
                    xbeg = np.random.uniform(x_lb, x_ub)
                    ybeg = np.random.uniform(y_lb, y_ub)
                    continue
                found = True
        else:
            xcond = (data[:, 0] <= xbeg + block_size) & (data[:, 0] >= xbeg)
            ycond = (data[:, 1] <= ybeg + block_size) & (data[:, 1] >= ybeg)
            cond = xcond & ycond
            if np.sum(cond) < 1000:  # discard block if there are less than 100 pts.
                continue

        block_data = data[cond, :]
        block_label = label[cond]

        if use_all_points:
            block_data_list.append(block_data)
            block_label_list.append(block_label)
        else:
            # randomly subsample data
            block_data_sampled, block_label_sampled = \
                sample_data_label(block_data, block_label, num_point)
            block_data_list.append(np.expand_dims(block_data_sampled, 0))
            block_label_list.append(np.expand_dims(block_label_sampled, 0))

    if use_all_points:
        block_data_return, block_label_return = np.array(block_data_list, dtype = object), np.array(block_label_list, dtype = object)
    else:
        block_data_return, block_label_return = np.concatenate(block_data_list, 0), np.concatenate(block_label_list, 0)

    return block_data_return, block_label_return