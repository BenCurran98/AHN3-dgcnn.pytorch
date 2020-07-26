import numpy as np
import glob

datapath = 'D:/Documents/Datasets/AHN3_as_S3DIS_RGB_NPY/'
for room_file in glob.iglob(datapath + '*.npy'):
    room_data = np.load(room_file)
    xyz_min = np.amin(room_data, axis=0)[0:3]
    xyz_max = np.amax(room_data, axis=0)[0:3]
    print(room_file + '    ' + str(room_data.shape[0]))
