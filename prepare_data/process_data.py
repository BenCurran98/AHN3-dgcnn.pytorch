import os
import glob
from shutil import Error
import numpy as np
import json
import h5py
import gc
import laspy
from tqdm import tqdm
import pointcloud_util as utils
from dtm import build_dtm, gen_agl
from sklearn.neighbors import KDTree

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
CLASS_MAP_FILE = os.path.join(ROOT_DIR, "params", "class_map.json")

def load_h5_pointcloud(filename, features_output = [], features = {}):
    """Load a pointcloud in the HDF5 format

    Args:
        filename (str): Name of point cloud file
        features_output (list, optional): List of point cloud features to extract. Defaults to [].
        features (dict, optional): Maps point feature name to column it occupies in the data matrix output. Defaults to {}.
    """
    file = h5py.File(filename, 'r+')
    # only really need position and classification - if we have agl, substitute for z
    
    features_output = [f for f in features_output if f in features.keys()]
    position = file['LAS/Position']
    data = np.zeros((position.shape[0], len(features_output)))
    if 'AGL' in file.keys() and "agl" in features_output:
        agl = file['AGL']
        data[:, features["agl"]] = agl

    labels = file['LAS/Classification']

    if "color" in features_output:
        data[:, features["color"]] = file["LAS/Color"]
    if "intensity" in features_output:
        data[:, features["intensity"]] = file["LAS/Intensity"]
    if "return_number" in features_output:
        data[:, features["return_number"]] = file["LAS/ReturnNumber"]
    if "number_of_returns" in features_output:
        data[:, features["number_of_returns"]] = file["LAS/NumberOfReturns"]

    return data, labels

def load_las_pointcloud(filename, features_output = [], features = {}):
    """Load a pointcloud in the LAS format

    Args:
        filename (str): Name of point cloud file
        features_output (list, optional): List of point cloud features to extract. Defaults to [].
        features (dict, optional): Maps point feature name to column it occupies in the data matrix output. Defaults to {}.
    """
    file = laspy.read(filename)

    avail_fields = [f.name for f in file.header.point_format]

    features_output = [f for f in features_output if f in features.keys() and 
                            (f in avail_fields or 
                                f.lower() in avail_fields or 
                                f.upper() in avail_fields
                                or f == "agl")]

    if any(["x" not in features_output, 
            "y" not in features_output, 
            "z" not in features_output,
            "x" not in features.keys(), 
            "y" not in features.keys(), 
            "z" not in features.keys()]):
        raise Error("No position found in pointcloud!")

    data = np.zeros((file.x.shape[0], len(features_output)))
    data[:, features["x"]] = file.x
    data[:, features["y"]] = file.y
    data[:, features["z"]] = file.z
    labels = file.classification

    
    if "red" in features_output:
        data[:, features["red"]] = file.red
    if "green" in features_output:
        data[:, features["green"]] = file.green
    if "blue" in features_output:
        data[:, features["blue"]] = file.blue
    if "intensity" in features_output:
        data[:, features["intensity"]] = file.intensity
    if "return_number" in features_output:
        data[:, features["return_number"]] = np.array(file.return_number)
    if "number_of_returns" in features_output:
        data[:, features["number_of_returns"]] = np.array(file.number_of_returns)

    return data, labels

def load_pointcloud(filename, features_output = [], features = {}, filter_noise = True):
    """Load a pointcloud from a file

    Args:
        filename (str): Name of pointcloud file
        features_output (list, optional): List of point features to return. Defaults to [].
        features (dict, optional): Maps point feature names to column index they will appear in in output matrix. Defaults to {}.
        filter_noise (bool, optional): Whether to filter noise. Defaults to True.

    Raises:
        Exception: Unsupported file type entered
    """
    if filename.split('.')[-1] == 'h5':
        data, labels = load_h5_pointcloud(filename, features_output = features_output, features = features)
    elif filename.split('.')[-1] == 'las':
        data, labels = load_las_pointcloud(filename, features_output = features_output, features = features)
    else:
        raise Exception('Unsupported file type!')

    if filter_noise:
        kdtree = KDTree(data[:, 0:3], metric = "euclidean")
        dists, _ = kdtree.query(data[:, 0:3], k = 2)
        good_idxs = np.where(dists[:, 1] < 1.0)[0]
        print("Filtered {} noise points".format(data.shape[0] - len(good_idxs)))
        data = data[good_idxs, :]
        labels = labels[good_idxs]

    return data, labels

def save_las_pointcloud(data, labels, filename, features_output = [], features = {}):
    """Save a pointcloud into LAS format

    Args:
        data (array-like): Array of pointcloud data
        labels (array-like): Array of pointcloud classification labels
        filename (str): Name of pointcloud file to save to
        features_output (list, optional): List of point cloud features to extract. Defaults to [].
        features (dict, optional): Maps point feature name to column it occupies in the data matrix output. Defaults to {}.
    """
    las = laspy.create(file_version = "1.2", point_format = 3) 

    las.x = data[:, 0]
    las.y = data[:, 1]
    las.z = data[:, 2]
    las.classification = labels.reshape(-1)

    features_output = [f for f in features_output if f in features.keys()]


    if "red" in features_output:
        las.red = data[:, features["red"]]
    if "green" in features_output:
        las.green = data[:, features["green"]]
    if "blue" in features_output:
        las.blue = data[:, features["blue"]]
    if "intensity" in features_output:
        las.intensity = data[:, features["intensity"]]
    if "return_number" in features_output:
        las.return_number = data[:, features["return_number"]]
    if "number_of_returns" in features_output:
        las.number_of_returns = data[:, features["number_of_returns"]]
            
    las.write(filename)
    

def load_pointcloud_dir(dir, outdir, 
                        block_size = 100, 
                        sample_num = 5, 
                        class_map_file = CLASS_MAP_FILE, 
                        min_num = 100, 
                        las_dir = "converted-pcs",
                        features_output = [],
                        features = {},
                        calc_agl = True,
                        cell_size = 1, 
                        desired_seed_cell_size = 90, 
                        boundary_block_width = 5, 
                        detect_water = False, 
                        remove_buildings = True, 
                        output_tin_file_path = None,
                        dtm_buffer = 6,
                        dtm_module_path = "",
                        num_points = 7000,
                        sub_block_size = 30,
                        use_all_points = False,
                        sub_sample_num = 10,
                        n_tries = 10):
    """Load pointcloud data into batches from a directory

    Args:
        dir (str): Name of directory to load from
        outdir (str): Namer of directory to store data files to
        block_size (int, optional): Size of initial blocks to split point clouds into. Defaults to 100.
        sample_num (int, optional): How many larger blocks to sample from each point cloud. Defaults to 5.
        class_map_file (str, optional): Name of JSON file containing class label mappings. Defaults to CLASS_MAP_FILE.
        min_num (int, optional): Minimum number of classes that must be present in each larger block to be stored. Defaults to 100.
        las_dir (str, optional): Directory to store LAS files of sampled blocks to. Defaults to "converted-pcs".
        features_output (list, optional): List of point features to store from each point cloud. Defaults to [].
        features (dict, optional): Maps point feature names to column indices where they will appear in data matrices. Defaults to {}.
        calc_agl (bool, optional): Whether to calculate AGL. Defaults to True.
        cell_size (int, optional): Size of cells used for DTM calculation. Defaults to 1.
        desired_seed_cell_size (int, optional): Expected seed cell size for DTM generation. Defaults to 90.
        boundary_block_width (int, optional): Width of blocks to put on boundary for DTM generation. Defaults to 5.
        detect_water (bool, optional): Whether to detect water in DTM generation. Defaults to False.
        remove_buildings (bool, optional): Whether to remove buildings in DTM generation. Defaults to True.
        output_tin_file_path (str, optional): Path to save DTM tin file to. Defaults to None.
        dtm_buffer (int, optional): Buffer width used in DTM generation (number of cells). Defaults to 6.
        dtm_module_path (str, optional): Location of the DTM generation module. Defaults to "".
        num_points (int, optional): Number of points to subsample in each smaller tile. Defaults to 7000.
        sub_block_size (int, optional): Size of smaller tiles to sample from larger tiles in a pointcloud. Defaults to 30.
        use_all_points (bool, optional): Whether to use all points in each sub tile. Defaults to False.
        sub_sample_num (int, optional): Number of sub block samples to take per tile. Defaults to 10.
        n_tries (int, optional): Number of attempts to search a tile for a suitable set of sub blocks. Defaults to 10.

    Returns:
        data_batches: List of batched point cloud data containing sets of sampled sub-tiles
        label_batches: List of batches point cloud classifications for sub tiles
    """
    with open(class_map_file, "r") as f:
        class_map = json.load(f)
    class_map = {int(k): v for (k, v) in class_map.items()}
    classes = [k for k in np.unique(list(class_map.values()))]

    print("CLASSES: ", classes)

    data_batch_list, label_batch_list = [], []
    files = os.listdir(dir)
    acceptable_files = [f for f in files if f.split('.')[-1] in ['h5', 'las']]

    tile_num = 0

    if not os.path.isdir(las_dir):
        os.mkdir(las_dir)

    for i in tqdm(range(len(acceptable_files)), desc = "Loading PCs"):
        f = acceptable_files[i]
        whole_data, whole_labels = load_pointcloud(os.path.join(dir, f), features_output = features_output, features = features)

        data, labels = utils.room2blocks(whole_data, 
                                            whole_labels, 
                                            100000, 
                                            block_size = block_size, 
                                            random_sample = False, 
                                            stride = block_size/2, 
                                            sample_num = sample_num, 
                                            use_all_points = True)

        num_good = 0
        with tqdm(range(data.shape[0]), desc = "Saving Data") as t:
            for i in range(data.shape[0]):
                this_data, this_labels = convert_pc_labels(data[i], 
                                                        labels[i], 
                                                        class_map_file = class_map_file)

                if calc_agl and "agl" in features_output and "agl" in features.keys():
                    dtm = build_dtm(this_data, 
                                    module_path = dtm_module_path,
                                    cell_size = cell_size,
                                    desired_seed_cell_size = desired_seed_cell_size,
                                    boundary_block_width = boundary_block_width,
                                    detect_water = detect_water,
                                    remove_buildings = remove_buildings,
                                    output_tin_file_path = output_tin_file_path,
                                    dtm_buffer = dtm_buffer)

                    agl = gen_agl(dtm, this_data)

                    this_data[:, features["agl"]] = agl

                found = 0
                n = 0
                while found < sample_num:
                    block_points, block_labels = utils.room2blocks(this_data, this_labels, num_points, block_size=sub_block_size,
                                                            stride=sub_block_size/2, random_sample=True, sample_num=sub_sample_num - found, use_all_points=use_all_points)
                    for i in range(len(block_points)):
                        this_block_points = block_points[i]
                        this_block_labels = block_labels[i]
                        label_counts = [len(np.where(this_block_labels == c)[0]) for c in classes]
                        if all([c > min_num * ((sub_block_size ** 2)/(block_size ** 2)) for c in label_counts]):
                            found += 1
                            las = laspy.create(file_version = "1.2", point_format = 3) 

                            las.x = this_block_points[:, 0].astype(float)
                            las.y = this_block_points[:, 1].astype(float)
                            current_idx = 2 if not calc_agl else 3
                            las.z = this_block_points[:, current_idx].astype(float)
                            current_idx += 1
                            las.classification = this_block_labels
                            if "red" in features_output:
                                las.red = this_block_points[:, current_idx]
                            if "green" in features_output:
                                las.green = this_block_points[:, current_idx + 1]
                            if "blue" in features_output:
                                las.blue = this_block_points[:, current_idx + 2]
                                current_idx += 3
                            if "intensity" in features_output:
                                las.intensity = this_block_points[:, current_idx]
                                current_idx += 1
                            if "return_number" in features_output:
                                las.return_number = this_block_points[:, current_idx]
                                current_idx += 1
                            if "number_of_returns" in features_output:
                                las.number_of_returns = this_block_points[:, current_idx]
                                current_idx += 1
                            las.write(os.path.join(las_dir, "Area_{}.las".format(tile_num)))
                            np.savetxt(os.path.join(outdir, 'Area_{}.txt'.format(
                                        tile_num)), np.hstack((this_block_points, 
                                        np.reshape(this_block_labels, 
                                                (len(this_block_labels), 1)))))
                
                            data_batch_list.append(this_block_points)
                            label_batch_list.append(this_block_labels)
                            tile_num += 1
                            num_good += 1
                    n += 1

                    if n > n_tries:
                        break
                
                t.update()
                gc.collect()
    
    data_batches = np.concatenate(data_batch_list, 0)
    label_batches = np.concatenate(label_batch_list, 0)
    return data_batches, label_batches

def convert_pc_labels(data, labels, class_map_file = CLASS_MAP_FILE):
    """Convert labels in a pointcloud to a format compatible with DGCNN

    Args:
        data (List): List of batched pointcloud data
        labels (List): List of batched pointcloud labels
        class_map_file (str): Name of file containing class mappings

    Returns:
        data: All valid entries of batch data
        labels: Remapped labels of batched data 
    """
    with open(class_map_file, "r") as f:
        class_map = json.load(f)

    class_map = {int(k): v for (k, v) in class_map.items()}

    valid_idxs = [i for i in range(len(labels)) if labels[i] in class_map.keys()]
    data = data[valid_idxs, :]
    labels = labels[valid_idxs]

    m = 1
    for c in class_map.keys():
        labels[np.where(labels == c)] = class_map[c]
        m += 1

    return data, labels

def extract_annotations(area, data_folder, output_path, categories, features, features_output):
    """Extract the labelled point clouds from a directory

    Args:
        area (str): Name of the area folder being processed
        data_folder (str): Folder containing data being processed
        output_path (str): Where to save the processed data
        categories (List): Valid class labels to be processed
        features (Dict): Dict mapping point feature name to the column index of that feature in the text data
        features_output (List): Point features being saved after processing
    """
    if not os.path.exists(output_path):
        os.mkdir(output_path)

    orig_output_path = output_path
    if not os.path.exists(output_path):
        os.mkdir(output_path)

    room_files = list(glob.iglob(os.path.join(data_folder, '*.txt')))
    for i in tqdm(range(len(room_files)), desc = "Extracting PC Data"):  # read each tiled point cloud
        room_id = i + 1
        room_file = room_files[i]
        output_path = orig_output_path
        output_path = os.path.join(output_path, 'Area_' + str(room_id))
        if not os.path.exists(output_path):
            os.mkdir(output_path)

        output_path = os.path.join(output_path, area)
        if not os.path.exists(output_path):
            os.mkdir(output_path)

        # load data
        room_data = np.loadtxt(room_file)
        output_label = room_data[:, -1]
        output_data = np.zeros((room_data.shape[0], len(features_output)))
        test = np.unique(output_label)
        # select the output features
        for feature_id, feature in enumerate(features_output):
            output_data[:, feature_id] = room_data[:, features[feature]]
        fmt = ["%.3f" for _ in range(output_data.shape[1])]
        with open(output_path + '/' + area + '_' + str(room_id) + '.txt', 'w+') as fout1:
            np.savetxt(fout1, output_data, fmt=fmt)
        fout1.close()

        # write file according to classes
        ANNO_PATH = os.path.join(output_path, 'Annotations')
        if not os.path.exists(ANNO_PATH):
            os.mkdir(ANNO_PATH)

        eff_categories = np.unique(output_label)
        print(eff_categories)
        for category in eff_categories:
            # find corresponding classes
            category_indices = np.where(output_label == category)[0]
            fmt = ["%.3f" for _ in range(output_data.shape[1])]
            with open(ANNO_PATH + '/' + categories[category] + '.txt', 'w+') as fout2:
                np.savetxt(fout2, output_data[category_indices, :], fmt=fmt)
            fout2.close()

def write_anno_paths(base_dir, root_dir):
    """Write the paths of the point cloud annotation files to a common location

    Args:
        base_dir (str): Base directory of the data
        root_dir (str): Root directory of the files
    """
    with open(os.path.join(root_dir, 'meta/anno_paths.txt'), 'w+') as anno_paths:
        paths = []
        for path in glob.glob(base_dir + '/processed/*/*/Annotations'):
            path = path.replace('\\', '/')
            paths.append(path)
        paths = np.array(paths)
        np.savetxt(anno_paths, paths, fmt='%s')
    anno_paths.close()

def collect_3d_data(root_dir, output_folder):
    """Collect data in the meta data folder into numpy files

    Args:
        root_dir (str): Root directory of the files
        output_folder (str): Where to save the numpy data files
    """
    anno_paths = [line.rstrip() for line in open(os.path.join(root_dir, 'meta/anno_paths.txt'))]

    if not os.path.exists(output_folder):
        os.mkdir(output_folder)

    for anno_path in tqdm(anno_paths):
        elements = anno_path.split('/')
        out_filename = elements[-3] + '_' + elements[-2] + '.npy'
        utils.collect_point_label(anno_path, os.path.join(output_folder, out_filename), 'numpy')

def write_npy_file_names(root_dir, data_path):
    """Save the names of the numpy data files for future reference

    Args:
        root_dir (str): Directory of the files
        data_path (str): Directory of the data files being referenced
    """
    with open(os.path.join(root_dir, 'meta/all_data_label.txt'), 'w+') as f:
        files = []
        for file in glob.iglob(data_path + '/*.npy'):
            file = os.path.basename(file)
            files.append(file)
        paths = np.array(files)
        np.savetxt(f, files, fmt='%s')
    f.close()

def process_data(base_dir, root_folder, pc_folder, data_folder, 
                processed_data_folder, npy_data_folder, area, categories_file, 
                features_file, features_output, block_size, sample_num, 
                min_class_num, class_map_file, calc_agl, cell_size, 
                desired_seed_cell_size, boundary_block_width, detect_water,
                remove_buildings, output_tin_file_path, dtm_buffer,
                dtm_module_path, num_points, sub_block_size, use_all_points,
                sub_sample_num, n_tries):
    """Pre-process raw point cloud data for the classifier

    Args:
        base_dir (str): Base directory of the data
        root_folder (str): Root directory of the files
        pc_folder (str): Name of directory to load point clouds from
        data_folder (st): Name of directory to save extracted sub tiles to
        processed_data_folder (str): Name of directory to store processed tile data
        npy_data_folder (str): Name of directory to store .npy data binary files to
        area (str): Label to assign to the set of data being processed
        categories_file (str): Name of JSON file that maps class labels to class names
        features_file (str): Name of JSON file mapping point feature names to column index in which they will appear in the data matrix
        features_output (list, optional): List of point features to store from each point cloud.
        block_size (int, optional): Size of initial blocks to split point clouds into
        sample_num (int, optional): How many larger blocks to sample from each point cloud
        min_class_num (int, optional): Minimum number of classes that must be present in each larger block to be stored
        class_map_file (str, optional): Name of JSON file containing class label mappings
        calc_agl (bool, optional): Whether to calculate AGL
        cell_size (int, optional): Size of cells used for DTM calculation
        desired_seed_cell_size (int, optional): Expected seed cell size for DTM generation
        boundary_block_width (int, optional): Width of blocks to put on boundary for DTM generation
        detect_water (bool, optional): Whether to detect water in DTM generation
        remove_buildings (bool, optional): Whether to remove buildings in DTM generation
        output_tin_file_path (str, optional): Path to save DTM tin file to
        dtm_buffer (int, optional): Buffer width used in DTM generation (number of cells)
        dtm_module_path (str, optional): Location of the DTM generation module
        num_points (int, optional): Number of points to subsample in each smaller tile
        sub_block_size (int, optional): Size of smaller tiles to sample from larger tiles in a pointcloud
        use_all_points (bool, optional): Whether to use all points in each sub tile
        sub_sample_num (int, optional): Number of sub block samples to take per tile
        n_tries (int, optional): Number of attempts to search a tile for a suitable set of sub blocks
    """
    with open(categories_file, 'r') as f:
        categories = json.load(f)

    with open(features_file, 'r') as f:
        features = json.load(f)

    if not os.path.isdir(base_dir):
        os.mkdir(base_dir)

    if os.path.isdir(data_folder):
        os.rmdir(data_folder)
        os.mkdir(data_folder)
    else:
        os.mkdir(data_folder)

    categories = {float(c): categories[c] for c in categories.keys()}

    print("Base: ", base_dir)
    print("Root: ", root_folder)
    print("Pc: ", pc_folder)
    print("Data: ", data_folder)
    print("Processed: ", processed_data_folder)
    print("NPY: ", npy_data_folder)

    print("Loading pointcloud data")
    load_pointcloud_dir(pc_folder, data_folder, 
                            block_size = block_size, 
                            sample_num = sample_num, 
                            min_num = min_class_num, 
                            class_map_file = class_map_file,
                            features_output = features_output,
                            features = features,
                            calc_agl = calc_agl, 
                            cell_size = cell_size, 
                            desired_seed_cell_size = desired_seed_cell_size, 
                            boundary_block_width = boundary_block_width, 
                            detect_water = detect_water,
                            remove_buildings = remove_buildings, 
                            output_tin_file_path = output_tin_file_path, 
                            dtm_buffer = dtm_buffer,
                            dtm_module_path = dtm_module_path,
                            num_points = num_points,
                            sub_block_size = sub_block_size,
                            use_all_points = use_all_points,
                            sub_sample_num = sub_sample_num,
                            n_tries = n_tries)
    print("Extracting annotations...")
    extract_annotations(area, data_folder, processed_data_folder, categories, 
                        features, features_output)
    print("Writing annotation paths...")
    write_anno_paths(base_dir, root_folder)
    print("collecting NPY data...")
    collect_3d_data(root_folder, npy_data_folder)
    print("Writitng NPY data...")
    write_npy_file_names(root_folder, npy_data_folder)
