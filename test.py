import os
from prepare_data.dtm import build_dtm, gen_agl
from prepare_data.pointcloud_util import room2blocks
import torch
from data import FugroDataset_eval, pc_collate_test
from model import DGCNN
import numpy as np
from torch.utils.data import DataLoader
from util import *
from prepare_data.process_data import load_pointcloud, save_las_pointcloud
import sklearn.metrics as metrics
from torch.utils.tensorboard import SummaryWriter
from torch.nn.functional import softmax
from tqdm import tqdm
from dtm import *

UNCLASSIFIED = 31

def test(k, io, 
            data_dir = "/media/ben/ExtraStorage/InnovationConference/Datasets/data_as_S3DIS_NRI_NPY",
            cell_size = 0.4641588833612779,
            block_size = 30.0,
            num_classes = 5,
            num_features = 4,
            test_batch_size = 8,
            dropout = 0.5,
            emb_dims = 1024,
            use_all_points = False,
            cuda = False,
            min_class_confidence = 0.8,
            model_label = "dgcnn_model",
            num_threads = 8,
            num_interop_threads = 2,
            model_root = "checkpoints/dgcnn",
            pred_dir = "predict"):
    """Perform inference using a DGCNN model

    Args:
        k (int): Number of neighbours to calculate in feature spaces
        io (IOStream): Stream where log data is sent to
        data_dir (str, optional): Directory containing the dataset in NPY format. Defaults to "/media/ben/ExtraStorage/InnovationConference/Datasets/data_as_S3DIS_NRI_NPY".
        num_points (int, optional): Number of points to sample from each block. Defaults to 5000.
        block_size (float, optional): Size of blocks to sample from each tile. Defaults to 30.0.
        num_classes (int, optional): Number of classes to train on. Defaults to 5.
        test_batch_size (int, optional): Number of test samples in each batch. Defaults to 8.
        dropout (float, optional): Dropout probability for dropout layer in model. Defaults to 0.5.
        emb_dims (int, optional): Dimensions to embed the global feature space into. Defaults to 1024.
        use_all_points (bool, optional): Whether to use all points in a block or to subsample. Defaults to False.
        cuda (bool, optional): Whether to use CUDA device for training. Defaults to False.
        min_class_confidence (float, optional): Minimum confidence value for the model to label a point as belonging to a class
        model_label (str, optional): Label to assign to the model for IO purposes. Defaults to "dgcnn_model".
        num_threads (int, optional): Number of primary threads for PyTorch to use in training. Defaults to 8.
        num_interop_threads (int, optional): Number of secondary threads for PyTorch to use for training. Defaults to 2.
        model_root (str, optional): Directory containing saved model files. Defaults to "checkpoints/dgcnn".
        pred_dir (str, optional): Directory to output the predictions into
    """
    DUMP_DIR = pred_dir
    torch.set_num_threads(num_threads)
    torch.set_num_interop_threads(num_interop_threads)
    all_true_cls = []
    all_pred_cls = []
    all_true_seg = []
    all_pred_seg = []
    for test_area in [1]:
        test_area = str(test_area)
        if (test_area == 'all') or (test_area == test_area):
            dataset = FugroDataset_eval(split='test', data_root=data_dir, cell_size = cell_size,
                                   block_size=block_size, use_all_points=use_all_points)
            test_loader = DataLoader(dataset, batch_size=test_batch_size, collate_fn = pc_collate_test, shuffle=False, drop_last=False)

            room_idx = np.array(dataset.room_idxs)

            fout_data_label = []
            true_data_labels = []
            for room_id in np.unique(room_idx):
                out_data_label_filename = 'Area_%s_room_%d_pred_gt.txt' % (test_area, room_id)
                out_data_label_filename = os.path.join(DUMP_DIR, out_data_label_filename)
                out_true_label_filename = 'Area_%s_room_%d_true_labels.txt' % (test_area, room_id)
                out_true_label_filename = os.path.join(DUMP_DIR, out_true_label_filename)
                if os.path.isfile(out_data_label_filename):
                    os.remove(out_data_label_filename)
                if os.path.isfile(out_true_label_filename):
                    os.remove(out_true_label_filename)
                fout_data_label.append(open(out_data_label_filename, 'a'))
                true_data_labels.append(open(out_true_label_filename, 'a'))

            device = torch.device("cuda" if cuda else "cpu")

            io.cprint('Start overall evaluation...')

            model = DGCNN(num_classes, num_features, k, dropout = dropout, emb_dims = emb_dims, cuda = cuda)

            if cuda:
                checkpoint = torch.load(os.path.join(model_root, '%s.t7' % model_label))
            else:
                checkpoint = torch.load(os.path.join(model_root, '%s.t7' % model_label), map_location=torch.device('cpu'))
            
            print("model: ", os.path.join(model_root, '%s.t7' % model_label))
            model.load_state_dict(checkpoint['model_state_dict'])
            model = model.eval()

            count_parameters(model)

            io.cprint('%s.t7 restored.' % model_label)

            test_acc = 0.0
            count = 0.0
            test_true_cls = []
            test_pred_cls = []
            test_true_seg = []
            test_pred_seg = []

            io.cprint('Start testing ...')
            num_batch = 0
            
            with tqdm(test_loader, desc = "Testing") as t:
                for data, seg in test_loader:
                    data, seg = data.to(device), seg.to(device)
                    data = data.permute(0, 2, 1).float()
                    batch_size = data.size()[0]

                    seg_pred, _ = model(data)
                    seg_pred = softmax(seg_pred, dim = 2)
                    seg_pred = seg_pred.permute(0, 2, 1).contiguous()
                    vals, pred = seg_pred.max(dim = 2)
                    # vals = seg_pred.amax(dim = 2)
                    pred[torch.where(vals < min_class_confidence)] = UNCLASSIFIED
                    seg_np = seg.cpu().numpy()
                    pred_np = pred.detach().cpu().numpy()
                    test_true_cls.append(seg_np.reshape(-1))
                    test_pred_cls.append(pred_np.reshape(-1))
                    test_true_seg.append(seg_np)
                    test_pred_seg.append(pred_np)

                    # write prediction results
                    for batch_id in range(batch_size):
                        pts = data[batch_id, :, :]
                        
                        pts = pts.permute(1, 0).float()
                        l = seg[batch_id, :]
                        pred_ = pred[batch_id, :]
                        logits = seg_pred[batch_id, :, :]
                        # compute room_id
                        room_id = room_idx[num_batch + batch_id]
                        for i in range(pts.shape[0]):
                            fout_data_label[room_id].write('%f %f %f %d\n' % (
                                pts[i, 0], pts[i, 1], pts[i, 2],
                                pred_[i]))

                            true_data_labels[room_id].write("%d\n" % l[i])
                    num_batch += batch_size
                    torch.cuda.empty_cache()

                    balanced_acc = metrics.balanced_accuracy_score(seg_np.reshape(-1), pred_np.reshape(-1))
                    acc = metrics.accuracy_score(seg_np.reshape(-1), pred_np.reshape(-1))
                    t.set_postfix(A = acc, BA = balanced_acc)
                    t.update()

            for room_id in np.unique(room_idx):
                fout_data_label[room_id].close()

            test_true_cls = np.concatenate(test_true_cls)
            test_pred_cls = np.concatenate(test_pred_cls)
            test_acc = metrics.accuracy_score(test_true_cls, test_pred_cls)
            avg_per_class_acc = metrics.balanced_accuracy_score(test_true_cls, test_pred_cls)
            
            outstr = 'Test :: test area: %s, test acc: %.6f, test avg acc: %.6f' % (test_area,
                                                                                    test_acc,
                                                                                    avg_per_class_acc,)
            io.cprint(outstr)

            # calculate confusion matrix
            conf_mat = metrics.confusion_matrix(test_true_cls, test_pred_cls)
            io.cprint('Confusion matrix:')
            io.cprint(str(conf_mat))

            all_true_cls.append(test_true_cls)
            all_pred_cls.append(test_pred_cls)
            all_true_seg.append(test_true_seg)
            all_pred_seg.append(test_pred_seg)

    if test_area == 'all':
        all_true_cls = np.concatenate(all_true_cls)
        all_pred_cls = np.concatenate(all_pred_cls)
        all_acc = metrics.accuracy_score(all_true_cls, all_pred_cls)
        avg_per_class_acc = metrics.balanced_accuracy_score(all_true_cls, all_pred_cls)
        all_true_seg = np.concatenate(all_true_seg, axis=0)
        all_pred_seg = np.concatenate(all_pred_seg, axis=0)
        all_ious = calculate_sem_IoU(all_pred_seg, all_true_seg, num_classes)
        outstr = 'Overall Test :: test acc: %.6f, test avg acc: %.6f, test iou: %.6f' % (all_acc,
                                                                                         avg_per_class_acc,
                                                                                         np.mean(all_ious))
        io.cprint(outstr)

def test_args(args, io):
    test(
        args.k,
        io,
        data_dir = args.data_dir,
        cell_size = args.cell_size,
        block_size = args.block_size,
        num_classes = args.num_classes,
        num_features = args.num_features,
        test_batch_size = args.test_batch_size,
        dropout = args.dropout,
        emb_dims = args.emb_dims,
        use_all_points = args.use_all_points,
        cuda = args.cuda,
        min_class_confidence = args.min_class_confidence,
        model_label = args.model_label, 
        num_threads = args.num_threads, 
        num_interop_threads = args.num_interop_threads,
        model_root = args.model_root,
        pred_dir = args.test_visu_dir
    )

def predict(k, io, pointcloud_file,
            pred_pointcloud_file,
            cell_size = 0.4641588833612779,
            block_size = 30.0,
            num_classes = 5,
            num_features = 4,
            dropout = 0.5,
            emb_dims = 1024,
            cuda = False,
            min_class_confidence = 0.8,
            model_label = "dgcnn_model",
            model_root = "checkpoints/dgcnn",
            features_output = [], 
            features = {}):
    
    model = DGCNN(num_classes, num_features, k, dropout = dropout, emb_dims = emb_dims, cuda = cuda)

    if cuda:
        checkpoint = torch.load(os.path.join(model_root, '%s.t7' % model_label))
    else:
        checkpoint = torch.load(os.path.join(model_root, '%s.t7' % model_label), map_location=torch.device('cpu'))

    print("model: ", os.path.join(model_root, '%s.t7' % model_label))
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.eval()

    count_parameters(model)

    io.cprint('%s.t7 restored.' % model_label)

    data, labels = load_pointcloud(pointcloud_file, features_output = features_output, features = features)

    print(data[0:10, :])

    dtm = build_dtm(data)
    agl = gen_agl(dtm, data)

    data = np.hstack((data, np.reshape(agl, (len(agl), 1))))

    print(data[0:10, :])

    block_data, _ = room2blocks(data, labels, cell_size = cell_size,
                                block_size = block_size,
                                stride = block_size,
                                random_sample =False,
                                use_all_points=False)

    preds = np.array([])
    data = []
    n = 1
    with tqdm(block_data, desc = "Classifying") as t:
        for X in block_data:
            np.savetxt("data{}.txt".format(n), X)
            if len(data) == 0:
                data = X
            else:
                data = np.vstack((data, X))

            x_lb = np.amin(X[:, 0])
            y_lb = np.amin(X[:, 1])

            X -= np.array([x_lb, y_lb, 0, 0])

            n += 1
            X = X[:, :, np.newaxis]
            X = torch.tensor(X)
            X = X.permute(2, 1, 0).float()
            print(X.shape)

            # seg_pred = model(data)
            # seg_pred = seg_pred.permute(0, 2, 1).contiguous()
            # seg_pred = softmax(seg_pred, dim = 2)
            # vals, pred = seg_pred.max(dim = 2)

            logit_pred, _ = model(X)
            logit_pred = logit_pred.permute(0, 2, 1).contiguous()
            logit_pred = softmax(logit_pred, dim = 2)
            print(logit_pred[0, 0:100, :])
            probs, pred = logit_pred.max(dim = 2)

            print(pred[0:100])
            print(probs[0:100, :])
            pred[torch.where(probs < min_class_confidence)] = UNCLASSIFIED
            pred = pred.detach().cpu().numpy()
            pred = pred.reshape(pred.shape[1], pred.shape[0])
            X = X.permute(2, 1, 0)
            X = X.detach().cpu().numpy()
            
            X = X[:, :, 0]

            X += np.array([x_lb, y_lb, 0, 0])
            
            if len(preds) == 0:
                preds = pred
            else:
                preds = np.vstack((preds, pred))

            save_las_pointcloud(X, pred, "pc_pred_{}.las".format(n))

            t.update()

    save_las_pointcloud(data, preds, pred_pointcloud_file)

    return data, preds