import os
import glob
from shutil import Error
import numpy as np
import json
import h5py
import gc
from numpy.core.shape_base import hstack
import laspy
from tqdm import tqdm
import argparse
import pointcloud_util as utils
from dtm import build_dtm, gen_agl
from sklearn.neighbors import NearestNeighbors, KDTree

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
CLASS_MAP_FILE = os.path.join(ROOT_DIR, "params", "class_map.json")

def load_h5_pointcloud(filename, features_output = [], features = {}):
    """Load a pointcloud in HDF5 format from `filename`"""
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

def load_las_pointcloud(filename, features_output = [], features = {}, filter_noise = True):
    """Load a pointcloud in LAS format from `filename`"""
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

    if filter_noise:
        kdtree = KDTree(data[:, 0:3], metric = "euclidean")
        dists, _ = kdtree.query(data[:, 0:3], k = 2)
        good_idxs = np.where(dists[:, 1] < 1)[0]
        print("Filtered {} noise points".format(data.shape[0] - len(good_idxs)))
        data = data[good_idxs, :]
        labels = labels[good_idxs]

    return data, labels

def load_pointcloud(filename, features_output = [], features = {}):
    """Load a pointcloud from a `filename"""
    if filename.split('.')[-1] == 'h5':
        return load_h5_pointcloud(filename, features_output = features_output, features = features)
    elif filename.split('.')[-1] == 'las':
        return load_las_pointcloud(filename, features_output = features_output, features = features)
    else:
        raise Exception('Unsupported file type!')

def save_las_pointcloud(data, labels, filename, features_output = [], features = {}):
    """Save a pointcloud in LAS format into `filename`
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
                        dtm_module_path = "/media/ben//ExternalStorage/external/RoamesDtmGenerator/bin"):
    """Load a set of pointclouds from a directory and save them in a txt file

    Args:
        dir (String): Directory the pointclouds are loaded from
        outdir (String): Directory the pointcloud text files are saved to

    Returns:
        data_batches: List of point data from pointclouds
        label_batches: List of label data from pointclouds
    """
    with open(class_map_file, "r") as f:
        class_map = json.load(f)
    class_map = {int(k): v for (k, v) in class_map.items()}
    classes = [k for k in class_map.values()]

    data_batch_list, label_batch_list = [], []
    files = os.listdir(dir)
    acceptable_files = [f for f in files if f.split('.')[-1] in ['h5', 'las']]

    tile_num = 0

    if not os.path.isdir(las_dir):
        os.mkdir(las_dir)

    print(features)
    print(features_output)

    for i in tqdm(range(len(acceptable_files)), desc = "Loading PCs"):
        f = acceptable_files[i]
        whole_data, whole_labels = load_pointcloud(os.path.join(dir, f), features_output = features_output, features = features)
        print(whole_data.shape)
        data, labels = utils.room2blocks(whole_data, 
                                            whole_labels, 
                                            10000, 
                                            block_size = block_size, 
                                            random_sample = False, 
                                            stride = block_size/2, 
                                            sample_num = sample_num, 
                                            use_all_points = True)

        print(data[0].shape)

        num_good = 0
        with tqdm(range(len(data)), desc = "Saving Data") as t:
            for i in range(len(data)):
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
                    # this_data = np.hstack((this_data, np.reshape(agl, (agl.shape[0], 1))))

                las = laspy.create(file_version = "1.2", point_format = 3) 

                las.x = this_data[:, 0]
                las.y = this_data[:, 1]
                current_idx = 2 if not calc_agl else 3
                las.z = this_data[:, current_idx]
                current_idx += 1
                las.classification = this_labels
                if "red" in features_output:
                    las.red = this_data[:, current_idx]
                if "green" in features_output:
                    las.green = this_data[:, current_idx + 1]
                if "blue" in features_output:
                    las.blue = this_data[:, current_idx + 2]
                    current_idx += 3
                if "intensity" in features_output:
                    las.intensity = this_data[:, current_idx]
                    current_idx += 1
                if "return_number" in features_output:
                    las.return_number = this_data[:, current_idx]
                    current_idx += 1
                if "number_of_returns" in features_output:
                    las.number_of_returns = this_data[:, current_idx]
                    current_idx += 1

                las.write(os.path.join(las_dir, "Area_{}.las".format(tile_num)))
                class_counts = [len(np.where(this_labels == c)[0]) for c in classes]
                if all([count > min_num for count in class_counts]):
                    np.savetxt(os.path.join(outdir, 'Area_{}.txt'.format(
                                            tile_num)), np.hstack((this_data, 
                                            np.reshape(this_labels, 
                                                    (len(this_labels), 1)))))
                    
                    data_batch_list.append(this_data)
                    label_batch_list.append(this_labels)
                    tile_num += 1
                    num_good += 1
                    t.set_postfix(num_good = num_good)
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
        area (String): Name of the area folder being processed
        data_folder (String): Folder containing data being processed
        output_path (String): Where to save the processed data
        categories (List): Valid class labels to be processed
        features (Dict): Dict mapping point feature name to index of that feature in the text data
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
        base_dir (String): Base directory of the data
        root_dir (String): Root directory of the files
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
        root_dir (String): Root directory of the files
        output_folder (String): Where to save the numpy data files
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
        root_dir (String): Directory of the files
        data_path (String): Directory of the data files being referenced
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
                dtm_module_path):
    """Pre-process raw data for the classifier

    Args:
        base_dir (String): Base directory of the data
        root_folder (String): Root folder of the files (contains meta-data directory)
        pc_folder (String): Folder containing pointcloud files
        data_folder (String): Folder containing extracted pointcloud data files
        processed_data_folder (String): Folder containing the complete datasets
        npy_data_folder (String): Output folder of the data summary
        area (String): Name of the folder containing the area data we want
        categories_file (String): JSON file containing label mappings
        features_file (String): JSON file containing index mappings of LiDAR features
        features_output (List): LiDAR features to extract
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
                            dtm_module_path = dtm_module_path)
    print("Extracting annotations...")
    extract_annotations(area, data_folder, processed_data_folder, categories, 
                        features, features_output)
    print("Writing annotation paths...")
    write_anno_paths(base_dir, root_folder)
    print("collecting NPY data...")
    collect_3d_data(root_folder, npy_data_folder)
    print("Writitng NPY data...")
    write_npy_file_names(root_folder, npy_data_folder)


if __name__ == "__main__":
    AREA = 'Training'
    PC_DIR = os.path.join(os.getcwd(), '..' '/Datasets', 'QualityTraining-orig')
    BASE_DIR = os.path.join(os.getcwd(), '..' '/Datasets')
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(description='Extract point cloud data')
    parser.add_argument('--base_dir', type = str, default = os.path.join(BASE_DIR, AREA), help = 'Base directory of data')
    parser.add_argument('--root_dir', type = str, default = ROOT_DIR, help = 'Root directory of the files')
    parser.add_argument('--area', type = str, default = AREA, help = 'Name of area to process')
    parser.add_argument('--pc_folder', type = str, default = PC_DIR)
    parser.add_argument('--data_folder', type = str, default = os.path.join(BASE_DIR, AREA, "Data"), help = 'Folder containing the complete datasets')
    parser.add_argument('--processed_data_folder', type = str, default = os.path.join(BASE_DIR, AREA, "processed"), help = 'Folder containing the complete datasets')
    parser.add_argument('--categories_file', type = str, default = 'params/categories.json', help = 'JSON file containing label mappings')
    parser.add_argument('--features_file', type = str, default = 'params/features.json', help = 'JSON file containing index mappings of LiDAR features')
    parser.add_argument('--class_map_file', type = str, default = CLASS_MAP_FILE, help = 'File containing class mappings')
    parser.add_argument('--features_output', nargs = '*', type = str, default = ["x", "y", "z", "agl"], help = 'LiDAR features to extract')
    parser.add_argument('--npy_data_folder', type = str, default = os.path.join(BASE_DIR, 'data_as_S3DIS_NRI_NPY'), help = 'Output folder of the data summary')
    parser.add_argument('--block_size', type = int, default = 100, help = 'Size of blocks to divide pointclouds into')
    parser.add_argument('--sample_num', type = int, default = 5, help = 'Number of tile samples to take from each point cloud')
    parser.add_argument('--min_class_num', type = int, default = 100, help = 'Minimum number of points per class for the pointcloud to be used')
    parser.add_argument('--calc_agl', type = bool, default = True, help = 'Whether to calculate AGL for the pointcloud')
    parser.add_argument('--cell_size', type = float, default = 1, help = 'Size of DTM cell')
    parser.add_argument('--desired_seed_cell_size', type = int, default = 90, help = 'Size of DTM seed cell')
    parser.add_argument('--boundary_block_width', type = int, default = 5, help = 'Number of blocks to use on the boundary')
    parser.add_argument('--detect_water', type = bool, default = False, help = 'Whether to detect water in DTM generation')
    parser.add_argument('--remove_buildings', type = bool, default = True, help = 'Whether to remove buildings in DTM generation')
    parser.add_argument('--output_tin_file_path', type = any, default = None, help = 'File path of the DTM tin file to produce')
    parser.add_argument('--dtm_buffer', type = float, default = 6, help = 'Buffer (metres) around the DTM region to use')
    parser.add_argument('--dtm_module_path', type = str, default = "/media/ben/ExtraStorage/external/RoamesDtmGenerator/bin", help = 'Path to the RoamesDTMGenerator module')
    
    
    args = parser.parse_args()
    
    process_data(args.base_dir, args.root_dir, args.pc_folder, args.data_folder, 
                args.processed_data_folder, args.npy_data_folder, args.area, 
                args.categories_file, args.features_file, args.features_output, 
                args.block_size, args.sample_num, args.min_class_num, args.class_map_file,
                args.calc_agl, args.cell_size, args.desired_seed_cell_size,
                args.boundary_block_width, args.detect_water,
                args.remove_buildings, args.output_tin_file_path, 
                args.dtm_buffer, args.dtm_module_path)