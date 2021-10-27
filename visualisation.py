#!/usr/bin/env python

import os
import torch
from data import FugroDataset_eval
from model import DGCNN
import numpy as np
import torch.nn.functional as F
from util import *
import laspy
from sklearn.cluster import DBSCAN
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from prepare_data.pointcloud_util import room2blocks

def generate_feature_map_clusters(model, x, depth, outdir = os.getcwd(), 
                                    outfile = "FeatureClusters.las", eps = 1.0, 
                                    min_samples = 10):
    """Cluster a point cloud based on proximity of points in the feature space of a DGCNN model at some depth in the network

    Args:
        model (DGCNN): Model performing inference
        x (tensor): Data being fed into the model
        depth (int): What layer to visualise (how many layers in)
        outdir (str, optional): Directory where plots are saved. Defaults to os.getcwd().
        outfile (str, optional): Name of LAS file to save clustered points to. Defaults to "FeatureClusters.las".
        eps (float, optional): DBSCAN search radius. Defaults to 1.0.
        min_samples (int, optional): Minimum number of neighbours needed for DBSCAN to create a cluster. Defaults to 10.
    """
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

def feature_tsne(x, depth, outdir = os.getcwd(),
                    outfile = "FeatureSpace.png", num_features = 4,
                    k = 40, num_classes = 4, dropout = 0.5, emb_dims = 1024, 
                    cuda = False, num_points = 7000, block_size = 30.0,
                    model_root = "",
                    model_label = "dgcnn_c4_k_40_agl",
                    class_colors = []):
    """Use T-Distributed Stochastic Nearest Neighbours (TSNE) to project and visualise the feature space at a given depth in the model into 2D

    Args:
        x (ndarray): Point cloud data being classified
        depth (int): Number of the layer in the network to be visualised
        outdir (str, optional): Directory to which the TSNE plots are saved to. Defaults to os.getcwd().
        outfile (str, optional): Name of the plot file to save. Defaults to "FeatureSpace.png".
        num_features (int, optional): Number of pointcloud features fed into the model. Defaults to 4.
        k (int, optional): Number of neighbours for the model to search for. Defaults to 40.
        num_classes (int, optional): Number of classes for the model to consider. Defaults to 4.
        dropout (float, optional): Dropout probability for the model. Defaults to 0.5.
        emb_dims (int, optional): Dimension that the model embeds global and local features into during inference. Defaults to 1024.
        cuda (bool, optional): Whether to send data to the GPU. Defaults to False.
        num_points (int, optional): Number of points to sample from each block. Defaults to 7000.
        block_size (float, optional): Size of the blocks the point cloud is tiled into. Defaults to 30.0.
        model_root (str, optional): Directory containing the model file to be loaded. Defaults to "".
        model_label (str, optional): Name of the model file to be loaded. Defaults to "dgcnn_c4_k_40_agl".
        class_colors (list, optional): Color values assigned to each class. Defaults to [].
    """
    if not os.path.isdir(outdir):
        os.makedirs(outdir)

    model = DGCNN(num_classes, num_features, k, dropout = dropout, emb_dims = emb_dims, cuda = cuda)

    if cuda:
        checkpoint = torch.load(os.path.join(model_root, '%s.t7' % model_label))
    else:
        checkpoint = torch.load(os.path.join(model_root, '%s.t7' % model_label), map_location=torch.device('cpu'))
    
    print("model: ", os.path.join(model_root, '%s.t7' % model_label))
    model.load_state_dict(checkpoint['model_state_dict'])

    count_parameters(model)
    model.eval()

    block_data, _ = room2blocks(x, np.ones(x.shape[0]), num_points,
                                block_size = block_size,
                                stride = block_size,
                                random_sample =False,
                                use_all_points=False)

    all_features = np.array([])
    n = 1
    for data in block_data:
        print("{}/{}".format(n, len(block_data)))
        np.savetxt("data_{}.txt".format(n), data)
        x_lb = np.amin(data[:, 0])
        y_lb = np.amin(data[:, 1])
        data -= np.array([x_lb, y_lb, 0, 0])

        data = data[:, :, np.newaxis]
        data = torch.tensor(data)
        data = data.permute(2, 1, 0).float()

        pred, features = model(data, depth = depth)

        if features.shape[2] == 1:
            continue

        features = features.permute(2, 1, 0)
        features = features.detach().cpu().numpy()

        pred = F.softmax(pred, dim = 2)
        pred = pred.permute(0, 2, 1).contiguous()
        _, pred = pred.max(dim = 2)
        pred = pred.detach().cpu().numpy()
        pred = pred.reshape(pred.shape[1], pred.shape[0])

        # calculate clusters using DBSCAN
        features = features.reshape(features.shape[0], features.shape[1])

        # now use TSNE to embed the feature space in 2D
        features_embedded = TSNE(n_components = 2, learning_rate = "auto", 
                                    init = "random").fit_transform(features)

        plt.figure()
        for this_class in np.unique(pred):
            if len(class_colors) < len(np.unique(pred)):
                color = np.random.random(3)
            else:
                color = class_colors[this_class]
            this_class_idxs = np.where(pred == this_class)[0]
            plt.scatter(features_embedded[this_class_idxs, 0], features_embedded[this_class_idxs, 1], color = color, label = "Class {}".format(this_class), s = 2)
        
        plt.legend(["Class {}".format(this_class) for this_class in np.unique(pred)])
        plt.title("Projected Feature Space Layer {}".format(depth))
        plt.savefig(os.path.join(outdir, outfile[0:-4] + "block{}_depth{}".format(n, depth) + outfile[-4:]))

        if len(all_features) == 0:
            all_features = features
        else:
            all_features = np.vstack((all_features, features))

        n += 1

def feature_tsne_all_layers(x, outdir = os.getcwd(),
                    outfile = "FeatureSpace.png", num_features = 4,
                    k = 40, num_classes = 4, dropout = 0.5, emb_dims = 1024, 
                    cuda = False, num_points = 7000, block_size = 30.0,
                    model_root = "",
                    model_label = "dgcnn_c4_k_40_agl"):
    """Use T-Distributed Stochastic Nearest Neighbours (TSNE) to project and visualise the feature space at each layer in the model into 2D

    Args:
        x (ndarray): Point cloud data being classified
        depth (int): Number of the layer in the network to be visualised
        outdir (str, optional): Directory to which the TSNE plots are saved to. Defaults to os.getcwd().
        outfile (str, optional): Name of the plot file to save. Defaults to "FeatureSpace.png".
        num_features (int, optional): Number of pointcloud features fed into the model. Defaults to 4.
        k (int, optional): Number of neighbours for the model to search for. Defaults to 40.
        num_classes (int, optional): Number of classes for the model to consider. Defaults to 4.
        dropout (float, optional): Dropout probability for the model. Defaults to 0.5.
        emb_dims (int, optional): Dimension that the model embeds global and local features into during inference. Defaults to 1024.
        cuda (bool, optional): Whether to send data to the GPU. Defaults to False.
        num_points (int, optional): Number of points to sample from each block. Defaults to 7000.
        block_size (float, optional): Size of the blocks the point cloud is tiled into. Defaults to 30.0.
        model_root (str, optional): Directory containing the model file to be loaded. Defaults to "".
        model_label (str, optional): Name of the model file to be loaded. Defaults to "dgcnn_c4_k_40_agl".
        class_colors (list, optional): Color values assigned to each class. Defaults to [].
    """
    
    class_colors = [(0.8, 0.0, 0.0), (0.33, 0.18, 0.0), (1.0, 0.5, 0.5), (0.56, 0.8, 0.2)]
    for depth in range(10, 11):
        this_outfile = outfile[0:-4] + str(depth)  + outfile[-4:]
        feature_tsne(x, depth = depth, outdir = outdir, outfile = this_outfile,
                        num_features = num_features,
                        k = k, num_classes = num_classes, dropout = dropout,
                        emb_dims = emb_dims, cuda = cuda, 
                        num_points = num_points, block_size = block_size,
                        model_root = model_root, model_label = model_label,
                        class_colors = class_colors)

def generate_feature_map_clusters(model, x, depth, outdir = os.getcwd(), 
                                    outfile = "FeatureClusters.las", eps = 1.0, 
                                    min_samples = 10):
    """Visualise the feature space of a DGCNN model at a given layer in the network in 2D

    Args:
        model (DGCNN): Model performing inference
        x (tensor): Data being fed into the model
        depth (int): What layer to visualise (how many layers in)
        outdir (str, optional): Directory where plots are saved. Defaults to os.getcwd().
        outfile (str, optional): Name of LAS file to save points clustered based on proximity in the feature space. Defaults to "FeatureClusters.las".
        eps (float, optional): DBSCAN search radius. Defaults to 1.0.
        min_samples (int, optional): Minimum number of neighbours needed for DBSCAN to create a cluster. Defaults to 10.
    """
    model.eval()
    # (batch_size, 3, num_points) -> (batch_size, feature_dims..., num_points)
    features = model(x, depth = depth)
    # calculate clusters using DBSCAN
    features = features.detach().numpy().reshape(features.size(2), features.size(1))

    clustering = DBSCAN(eps = np.percentile(eps, 0.5), min_samples = min_samples).fit(features)
    unique_labels = np.unique(clustering.labels_)
    print("Found {} clusters".format(max(unique_labels)))

    x = torch.cat([x[i, :, :] for i in range(x.size(0))]).numpy().transpose(1, 0)
    las = laspy.create(file_version = "1.2", point_format = 3)

    las.x = x[:, 0]
    las.y = x[:, 1]
    las.z = x[:, 2]
    las.pt_src_id = clustering.labels_

    las.write(os.path.join(outdir, outfile))
