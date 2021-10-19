import os
import sys
import tempfile
from shutil import rmtree, move
import numpy as np
import warnings
from subprocess import Popen, PIPE
from sklearn.neighbors import KDTree

def get_dtm_runner(module_path = "/media/ben/ExtraStorage/external/RoamesDtmGenerator/bin"):
    """Get the executable path for the RoamesDTMGenerator module
    """
    if sys.platform == "linux":
        dtm_runner = os.path.join(module_path, "DTMGeneration")
    elif sys.platform == "windows":
        dtm_runner = os.path.join(module_path, "DTMGeneration.exe")
    else:
        raise Exception("DTM Generator only supports Linux and Windows!")

    return dtm_runner

def build_dtm(pc, module_path = "/media/ben/ExtraStorage/external/RoamesDtmGenerator/bin",
            cell_size = 0.4641588833612779, desired_seed_cell_size = 100.0, 
            boundary_block_width = 5, detect_water = False, 
            remove_buildings = True, output_tin_file_path = None,
            dtm_buffer = 6, stdout_file = "", is_dry_run = False):
    """Build the DTM model for a point cloud

    Args:
        pc (ndarray): Pointcloud positions for the dtm
        module_path (str, optional): Path to DTM module. Defaults to "/home/ben/external/RoamesDtmGenerator/bin".
        cell_size (int, optional): Size of dtm cell. Defaults to 1.
        desired_seed_cell_size (int, optional): Size of seed cells. Defaults to 90.
        boundary_block_width (int, optional): Width of blocks on the boundary. Defaults to 5.
        detect_water (bool, optional): Whether to detect water. Defaults to False.
        remove_buildings (bool, optional): Whether to remove buildings. Defaults to True.
        output_tin_file_path ([type], optional): Path to the tin file generated for the dtm. Defaults to None.
        dtm_buffer (int, optional): Buffer around dtm boundary to generate. Defaults to 6.
        stdout_file (str, optional): File to write stdout contents from DTM module execution. Defaults to "".
        is_dry_run (bool, optional): Whether to just generate the binary DTM file. Defaults to False.
    """

    if pc.shape[0] == 0:
        warnings.warn("No points found in pointcloud!")
        return pc

    dtm_runner = get_dtm_runner(module_path = module_path)

    dump_tin_ply = output_tin_file_path != None

    temp_dir = tempfile.mkdtemp()
    
    min_x = np.amin(pc[:, 0])
    min_y = np.amin(pc[:, 1])
    max_x = np.amax(pc[:, 0])
    max_y = np.amax(pc[:, 1])

    pc_txt_file_path = os.path.join(temp_dir, "temp.txt")
    region_txt_file_path = os.path.join(temp_dir, "regions.txt")
    output_path = os.path.join(temp_dir,"output")
    if not os.path.isdir(output_path):
        os.makedirs(output_path, exist_ok = True)
    
    np.savetxt(pc_txt_file_path, pc)
    
    min_x = np.floor(min_x) - dtm_buffer
    max_x = np.ceil(max_x) + dtm_buffer

    min_y = np.floor(min_y) - dtm_buffer
    max_y = np.ceil(max_x) + dtm_buffer

    dtm_width = max(max_x - min_x, max_y - min_y)

    with open(region_txt_file_path, "w") as f:
        f.write("{} {} {} {}".format(min_x, 
                                    min_y, 
                                    min_x + dtm_width, 
                                    min_y + dtm_width))

    cmd = [
        dtm_runner, pc_txt_file_path, output_path, region_txt_file_path,
        str(cell_size), str(desired_seed_cell_size), str(boundary_block_width), 
        "true" if detect_water else "false",
        "true" if remove_buildings else "false",
        "true" if dump_tin_ply  else "false"
    ]

    p = Popen(cmd, stdout=PIPE, stderr=PIPE, stdin=PIPE)
    output = p.stdout.read()

    if stdout_file != "":
        with open(stdout_file, "wb") as f:
            f.write(output)

    if is_dry_run:
        return None
    new_dtm_file = os.path.join(output_path, 
                                "dtm_{}_{}.dat".format(int(min_x), int(min_y)))
    num_dtm_points = int(np.floor(dtm_width/cell_size + 1 * 0.5)  ** 2)
    dtm_edge_size = int(np.sqrt(num_dtm_points)) + 1
    if not os.path.isfile(new_dtm_file):
        warnings.warn("DTM Binary file notfound: {}".format(new_dtm_file))
        return pc - np.mean(pc)

    with open(new_dtm_file, "rb") as f:
        dtm_heights = np.fromfile(f, dtype = np.float32)
    
    points = np.zeros((dtm_edge_size * dtm_edge_size, 3))
    for row in range(dtm_edge_size):
        for col in range(dtm_edge_size): 
            points[row * dtm_edge_size + col, :] = [
                                        min_x + row * cell_size,
                                        min_y + col * cell_size,
                                        dtm_heights[col * dtm_edge_size + row]
                                                    ]
    
    no_data_value = -1e4
    not_nan_array = np.where(points[:, 2] != no_data_value)[0]

    points = points[not_nan_array]
    
    np.savetxt(os.path.join(temp_dir, "pts.txt"), points)

    if dump_tin_ply:
        tin_file_base = "tin_{}_{}.ply".format(int(min_x), int(min_y))
        tin_file_path = os.path.join(output_path, tin_file_base)
        if os.path.isfile(os.path.join(output_tin_file_path, tin_file_base)):
            os.remove(os.path.join(output_tin_file_path, tin_file_base))
        os.makedirs(output_tin_file_path, exist_ok = True)
        move(tin_file_path, output_tin_file_path)
    
    rmtree(temp_dir)

    return points

def gen_agl(dtm, pc):
    """Generate the agl for a pointcloud given a DTM

    Args:
        dtm (array_like): DTM points generated by DTM module
        pc (ndarray): Pointcloud positions to have AGL calculated

    Returns:
        agl: Vector of AGL values for all points in the pointcloud
    """
    dtm = np.array(dtm)
    if dtm.shape[0] == 0:
        return np.zeros(pc.shape[0])

    kdtree = KDTree(dtm[:, 0:2])

    k = min(4, dtm.shape[0])
    _, idxs = kdtree.query(pc[:, 0:2], k = k)
    avg_dtm_height = np.array([np.mean(dtm[idxs[i], 2]) for i in range(idxs.shape[0])])
    pc_agl = pc[:, 2] - avg_dtm_height

    return pc_agl