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
        anno_path: path to annotations. e.g. Area_1/office_2/Annotations/
        out_filename: path to save collected points and labels (each line is XYZRGBL)
        file_format: txt or numpy, determines what file format to save.
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
    """ data is in N x ...
        we want to keep num_samplexC of them.
        if N > num_sample, we will randomly keep num_sample of them.
        if N < num_sample, we will randomly duplicate samples.
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
    new_data, sample_indices = sample_data(data, num_sample)
    new_label = label[sample_indices]
    return new_data, new_label


def room2blocks(data, label, cell_size = 0.4641588833612779, block_size=1.0, stride=1.0,
                random_sample=False, sample_num=None, sample_aug=1, use_all_points=False):
    """ Prepare block training data.
    Args:
        data: N x 6 numpy array, 012 are XYZ in meters, 345 are RGB in [0,1]
            assumes the data is shifted (min point is origin) and aligned
            (aligned with XYZ axis)
        label: N size uint8 numpy array from 0-12
        num_point: int, how many points to sample in each block
        block_size: float, physical size of the block in meters
        stride: float, stride for block sweeping
        random_sample: bool, if True, we will randomly sample blocks in the room
        sample_num: int, if random sample, how many blocks to sample
            [default: room area]
        sample_aug: if random sample, how much aug
    Returns:
        block_datas: K x num_point x 6 np array of XYZRGB, RGB is in [0,1]
        block_labels: K x num_point x 1 np array of uint8 labels
        
    TODO: for this version, blocking is in fixed, non-overlapping pattern.
    """
    assert (stride <= block_size)

    x_ub = np.amax(data[:, 0])
    x_lb = np.amin(data[:, 0])
    
    y_ub = np.amax(data[:, 1])
    y_lb = np.amin(data[:, 1])

    z_ub = np.amax(data[:, 2])
    z_lb = np.amin(data[:, 2])

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
            sample_num = num_block_x * num_block_y * sample_aug
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
            # split into cells of size cell size, select points to get density
            sample_points = []
            sample_labels = []
            num_to_sample = int(np.ceil(1/(cell_size ** 3)))
            num_block_z = int(np.ceil(z_ub - z_lb)/cell_size + 1)
            num_blocks = int(np.ceil(block_size/cell_size))
            for i in range(num_blocks):
                for j in range(num_block_z):
                    this_zcond = (block_data[:, 2] <= z_lb + (j + 1) * cell_size) & (block_data[:, 2] >= z_lb + cell_size * j)
                    if len(np.where(this_zcond)[0]) == 0:
                        continue
                    this_xcond = (block_data[:, 0] <= xbeg + (i + 1) * cell_size) & (block_data[:, 0] >= xbeg + cell_size * i)
                    this_ycond = (block_data[:, 1] <= ybeg + (i + 1) * cell_size) & (block_data[:, 1] >= ybeg + cell_size * i)
                    
                    this_cond = this_xcond & this_ycond & this_zcond
                    these_points = block_data[this_cond, :]
                    these_labels = block_label[this_cond]

                    if these_points.shape[0] > 0:
                        these_points_sampled, these_labels_sampled = \
                            sample_data_label(these_points, these_labels, num_to_sample)
                        sample_points.append(these_points_sampled)
                        sample_labels.append(these_labels_sampled)
            if len(sample_points) > 0:
                block_data_sampled = np.concatenate(sample_points, 0)
                block_label_sampled = np.concatenate(sample_labels, 0)

                block_data_list.append(np.expand_dims(block_data_sampled, 0))
                block_label_list.append(np.expand_dims(block_label_sampled, 0))
    
    print([d.shape for d in block_data_list])
    return block_data_list, block_label_list
    # if use_all_points:
    #     block_data_return, block_label_return = np.array(block_data_list, dtype = object), np.array(block_label_list, dtype = object)
    # else:
    #     block_data_return, block_label_return = np.concatenate(block_data_list, 0), np.concatenate(block_label_list, 0)

    # return block_data_return, block_label_return