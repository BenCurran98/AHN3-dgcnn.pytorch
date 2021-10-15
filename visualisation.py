#!/usr/bin/env python

import os
import torch
from data import FugroDataset_eval
from model import DGCNN, knn
import numpy as np
from torch.utils.data import DataLoader
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

def feature_tsne(x, depth, outdir = os.getcwd(),
                    outfile = "FeatureSpace.png", num_features = 4,
                    k = 40, num_classes = 4, dropout = 0.5, emb_dims = 1024, 
                    cuda = False, num_points = 7000, block_size = 30.0,
                    model_root = "/media/ben/ExtraStorage/InnovationConference/FugroDGCNN/checkpoints/dgcnn_test_30epochs_p100/models/",
                    model_label = "dgcnn_c4_k_40_agl",
                    class_colors = []):
    if not os.path.isdir(outdir):
        os.makedirs(outdir)

    model = DGCNN(num_classes, num_features, k, dropout = dropout, emb_dims = emb_dims, cuda = cuda)

    if cuda:
        checkpoint = torch.load(os.path.join(model_root, '%s.t7' % model_label))
    else:
        checkpoint = torch.load(os.path.join(model_root, '%s.t7' % model_label), map_location=torch.device('cpu'))
    
    print("model: ", os.path.join(model_root, '%s.t7' % model_label))
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.eval()

    count_parameters(model)
    model.eval()
    # (batch_size, 3, num_points) -> (batch_size, feature_dims..., num_points)
    # x = x[:, :, np.newaxis]
    # print(x.shape)
    # x = torch.tensor(x)
    # x = x.permute(2, 1, 0)

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

        print("F0: ", features.shape)

        if features.shape[2] == 1:
            continue

        features = features.permute(2, 1, 0)
        features = features.detach().cpu().numpy()

        pred = F.softmax(pred, dim = 2)
        pred = pred.permute(0, 2, 1).contiguous()
        _, pred = pred.max(dim = 2)
        pred = pred.detach().cpu().numpy()
        pred = pred.reshape(pred.shape[1], pred.shape[0])

        print("F1: ", features.shape)
        # calculate clusters using DBSCAN
        features = features.reshape(features.shape[0], features.shape[1])
        print("F2: ", features.shape)

        # now use TSNE to embed the feature space in 2D
        features_embedded = TSNE(n_components = 2, learning_rate = "auto", 
                                    init = "random").fit_transform(features)

        print("FE: ", features_embedded.shape)
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


    # print("F1: ", all_features.shape)
    # # calculate clusters using DBSCAN
    # all_features = all_features.reshape(all_features.shape[0], all_features.shape[1])
    # print("F2: ", all_features.shape)

    # # now use TSNE to embed the feature space in 2D
    # features_embedded = TSNE(n_components = 2, learning_rate = "auto", 
    #                             init = "random").fit_transform(all_features)

    # print("FE: ", features_embedded.shape)
    # plt.figure()
    # for this_class in np.unique(pred):
    #     color = np.random.random(3)
    #     this_class_idxs = np.where(pred == this_class)[0]
    #     plt.scatter(features_embedded[this_class_idxs, 0], features_embedded[this_class_idxs, 1], color = color, label = "Class {}".format(this_class))
    
    # plt.legend(["Class {}".format(this_class) for this_class in np.unique(pred)])
    # plt.title("Projected Feature Space Layer {}".format(depth))
    # plt.savefig(os.path.join(outdir, outfile))

def feature_tsne_all_layers(x, outdir = os.getcwd(),
                    outfile = "FeatureSpace.png", num_features = 4,
                    k = 40, num_classes = 4, dropout = 0.5, emb_dims = 1024, 
                    cuda = False, num_points = 7000, block_size = 30.0,
                    model_root = "/media/ben/ExtraStorage/InnovationConference/FugroDGCNN/checkpoints/dgcnn_test_30epochs_p100/models/",
                    model_label = "dgcnn_c4_k_40_agl"):
    
    class_colors = [(0.8, 0.0, 0.0), (0.33, 0.18, 0.0), (1.0, 0.5, 0.5), (0.56, 0.8, 0.2)]
    for depth in range(10, 11):
        # this_outfile = outfile[0:-4] + str(depth)  + outfile[-4:]
        this_outfile = outfile
        feature_tsne(x, depth = depth, outdir = outdir, outfile = this_outfile,
                        k = k, num_classes = num_classes, dropout = dropout,
                        emb_dims = emb_dims, cuda = cuda, 
                        num_points = num_points, block_size = block_size,
                        model_root = model_root, model_label = model_label,
                        class_colors = class_colors)