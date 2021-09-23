#!/usr/bin/env python

import os
import torch
from data import FugroDataset_eval
from model import DGCNN, knn
import numpy as np
from torch.utils.data import DataLoader
from util import *
import laspy
from sklearn.cluster import DBSCAN

def generate_feature_map_clusters(model, x, depth, outdir = os.getcwd(), 
                                    outfile = "FeatureClusters.las", eps = 1.0, 
                                    min_samples = 10):
    model.eval()
    # (batch_size, 3, num_points) -> (batch_size, feature_dims..., num_points)
    features = model(x, depth = depth)
    # calculate clusters using DBSCAN
    features = features.detach().numpy().reshape(features.size(2), features.size(1))

    clustering = DBSCAN(eps = np.percentile(eps, 0.5), min_samples = min_samples).fit(features)
    unique_labels = np.unique(clustering.labels_)

    x = torch.cat([x[i, :, :] for i in range(x.size(0))]).numpy().transpose(1, 0)
    las = laspy.create(file_version = "1.2", point_format = 3)

    las.x = x[:, 0]
    las.y = x[:, 1]
    las.z = x[:, 2]
    las.pt_src_id = clustering.labels_

    las.write(os.path.join(outdir, outfile))
