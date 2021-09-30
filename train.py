from __future__ import print_function
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from data import FugroDataset
from data import S3DISDataset_eval
from model import DGCNN
import numpy as np
from torch.utils.data import DataLoader
from util import *
import sklearn.metrics as metrics
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

def train(k, io, 
            data_dir = "/media/ben/T7 Touch/InnovationConference/Datasets/data_as_S3DIS_NRI_NPY",
            num_points = 5000,
            block_size = 30.0,
            epochs = 30,
            num_classes = 5,
            num_features = 3,
            train_batch_size = 8,
            test_batch_size = 8,
            min_class_num = 100,
            use_sgd = False,
            lr = 0.001,
            momentum = 0.9,
            dropout = 0.5,
            emb_dims = 1024,
            sample_num = 5,
            scheduler = "cos",
            test_prop = 0.2,
            use_all_points = False,
            cuda = False,
            model_label = "dgcnn_model",
            num_threads = 8,
            num_interop_threads = 2,
            exclude_classes = [],
            model_root = "checkpoints/dgcnn",
            exp_name = "DGCNN_Training",
            tb_dir = "tensorboard_logs"):
    """Train a DGCNN model

    Args:
        k (int): Number of neighbours to calculate in feature spaces
        io (IOStream): Stream where log data is sent to
        data_dir (str, optional): Directory containing the dataset in NPY format. Defaults to "/media/ben/T7 Touch/InnovationConference/Datasets/data_as_S3DIS_NRI_NPY".
        num_points (int, optional): Number of points to sample from each block. Defaults to 5000.
        block_size (float, optional): Size of blocks to sample from each tile. Defaults to 30.0.
        epochs (int, optional): Number of epochs to train on. Defaults to 30.
        num_classes (int, optional): Number of classes to train on. Defaults to 5.
        train_batch_size (int, optional): Number of training samples in each batch. Defaults to 8.
        test_batch_size (int, optional): Number of test samples in each batch. Defaults to 8.
        min_class_num (int, optional): Minimum number of points per class for a block to be used. Defaults to 100.
        use_sgd (bool, optional): Indicates whether to use Stochastic Gradient Descent as an optimiser. Defaults to False.
        lr (float, optional): Learning rate of optimiser. Defaults to 0.001.
        momentum (float, optional): Momentum of optimiser (only used for SGD). Defaults to 0.9.
        dropout (float, optional): Dropout probability for dropout layer in model. Defaults to 0.5.
        emb_dims (int, optional): Dimensions to embed the global feature space into. Defaults to 1024.
        sample_num (int, optional): Number of blocks to sample from each tile. Defaults to 5.
        scheduler (str, optional): Schedules the adjustment of the learning rate. Defaults to "cos".
        test_prop (float, optional): Proportion of the dataset to use for testing/validation. Defaults to 0.2.
        use_all_points (bool, optional): Whether to use all points in a block or to subsample. Defaults to False.
        cuda (bool, optional): Whether to use CUDA device for training. Defaults to False.
        model_label (str, optional): Label to assign to the model for IO purposes. Defaults to "dgcnn_model".
        num_threads (int, optional): Number of primary threads for PyTorch to use in training. Defaults to 8.
        num_interop_threads (int, optional): Number of secondary threads for PyTorch to use for training. Defaults to 2.
        model_root (str, optional): Directory containing saved model files. Defaults to "checkpoints/dgcnn".
        exp_name (str, optional): Name of the training experiment. Defaults to "DGCNN_Training".
        tb_dir (str, optional): Directory containing tensorboard log files. Defaults to "tensorboard_logs".
    """
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    torch.set_num_threads(num_threads)
    torch.set_num_interop_threads(num_interop_threads)

    train_data = FugroDataset(split='train', data_root=data_dir, num_point=num_points,
                     block_size=block_size, use_all_points = use_all_points, test_prop = test_prop, sample_num = sample_num, class_min = min_class_num, classes = range(num_classes))
    train_loader = DataLoader(
        train_data, num_workers=8, batch_size=train_batch_size,
        shuffle=True, drop_last=True)

    test_data = FugroDataset(split='test', data_root=data_dir, num_point=num_points,
                     block_size=block_size, test_prop = test_prop, classes = range(num_classes))
    test_loader = DataLoader(
        test_data, num_workers=8, batch_size=test_batch_size,
        shuffle=True, drop_last=True)

    device = torch.device("cuda" if cuda else "cpu")
    print("Using ", device)

    model = DGCNN(num_classes, num_features, k, dropout = dropout, emb_dims = emb_dims, cuda = cuda)

    print(str(model))
    
    if cuda:
        print("Let's use", torch.cuda.device_count(), "GPUs!")

    if use_sgd:
        print("Use SGD")
        opt = optim.SGD(model.parameters(), lr=lr * 100, momentum=momentum, weight_decay=1e-4)
    else:
        print("Use Adam")
        opt = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    # cosine annealing is a good way to avoid gradient decay
    if scheduler == 'cos':
        scheduler = CosineAnnealingLR(opt, epochs, eta_min=1e-3)
    elif scheduler == 'step':
        scheduler = StepLR(opt, 20, 0.5, epochs)

    # check if we have a model saved somewhere- if so, load it and train
    try:
        checkpoint = torch.load(os.path.join(model_root, "{}.t7".format(model_label)))
        start_epoch = checkpoint['epoch'] + 1
        model.load_state_dict(checkpoint['model_state_dict'])
        opt.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        best_per_class_acc = checkpoint['mBPCA']
        io.cprint('Use pretrained model')
    except:
        io.cprint('No existing model, starting training from scratch...')
        start_epoch = 0
        best_per_class_acc = 0

    criterion = cal_loss

    log_dir = os.path.join(BASE_DIR, tb_dir)
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)

    writer_train_loss = SummaryWriter(os.path.join(log_dir, 'train_loss'))
    writer_train_accuracy = SummaryWriter(os.path.join(log_dir), 'train_accuracy')
    writer_train_balanced_accuracy = SummaryWriter(os.path.join(log_dir), 'balanced_train_accuracy')
    writer_test_accuracy = SummaryWriter(os.path.join(log_dir), 'test_accuracy')
    writer_test_balanced_accuracy = SummaryWriter(os.path.join(log_dir), 'balanced_test_accuracy')

    for epoch in range(start_epoch, epochs):
        ####################
        # Train
        ####################
        train_loss = 0.0
        count = 0.0
        niter = epoch * len(train_loader) * train_batch_size
        model.train()
        train_true_cls = []
        train_pred_cls = []
        train_true_seg = []
        train_pred_seg = []

        with tqdm(train_loader, desc = "Training epoch {}".format(epoch)) as t:
            for data, seg, idx in train_loader:
                mask = torch.tensor(train_data.create_train_mask(idx, data.shape[0], exclude_classes = exclude_classes))
                data, seg, mask = data.to(device), seg.to(device), mask.to(device)
                data = data.permute(0, 2, 1).float()
                batch_size = data.size()[0]
                opt.zero_grad()
                seg_pred = model(data) # batch_size * num_points * num_classes
                seg_pred = F.softmax(seg_pred, dim = 1)

                # only use the points indicated by the mask in back propogation- this acts as a kind of label balancing
                focus_seg = num_classes * torch.ones_like(seg)
                focus_pred = torch.zeros((seg_pred.shape[0], num_classes + 1, seg_pred.shape[2]))
                for i in range(mask.shape[0]):
                    masked_idxs = np.where(mask[i, :])[0]
                    focus_seg[i, masked_idxs] = seg[i, masked_idxs]
                    focus_pred[i, :, masked_idxs] = torch.cat((seg_pred[i, :, masked_idxs], torch.zeros(1, len(masked_idxs))), dim = 0)
                    for j in range(len(mask[i, :])):
                        if mask[i, j] == 0:
                            focus_pred[i, num_classes, j] = 1
                
                seg = focus_seg
                seg_pred = focus_pred
                seg_pred = seg_pred.permute(0, 2, 1).contiguous()

                loss = criterion(seg_pred.view(-1, num_classes + 1), seg.view(-1, 1).squeeze().long())
                loss.backward()
                opt.step()
                
                pred = seg_pred.max(dim=2)[1]  # (batch_size, num_points)
                count += batch_size
                train_loss += loss.item() * batch_size
                niter += batch_size
                writer_train_loss.add_scalar('Train/loss', loss.item(), niter)
                seg_np = seg.cpu().numpy()  # (batch_size, num_points)
                pred_np = pred.detach().cpu().numpy()  # (batch_size, num_points)
                train_true_cls.append(seg_np.reshape(-1))  # (batch_size * num_points)
                train_pred_cls.append(pred_np.reshape(-1))  # (batch_size * num_points)
                train_true_seg.append(seg_np)
                train_pred_seg.append(pred_np)
                
                # Report on both overall accuracy and label balanced accuracy
                true_labels = seg_np.reshape(-1)
                true_seg_idxs = np.where(true_labels != num_classes)[0]
                true_labels = true_labels[true_seg_idxs]
                pred_labels = pred_np.reshape(-1)
                pred_idxs = np.where(pred_labels != num_classes)[0]
                pred_labels = pred_labels[pred_idxs]
                balanced_train_acc = metrics.balanced_accuracy_score(true_labels, pred_labels)
                train_acc = metrics.accuracy_score(true_labels, pred_labels)
                t.set_postfix(A = train_acc, BA = balanced_train_acc)
                t.update()

        if scheduler == 'cos':
            scheduler.step()
        elif scheduler == 'step':
            if opt.param_groups[0]['lr'] > 1e-5:
                scheduler.step()
            if opt.param_groups[0]['lr'] < 1e-5:
                for param_group in opt.param_groups:
                    param_group['lr'] = 1e-5

        train_true_cls = np.concatenate(train_true_cls)
        train_pred_cls = np.concatenate(train_pred_cls)
        train_acc = metrics.accuracy_score(train_true_cls, train_pred_cls)
        balanced_train_acc = metrics.balanced_accuracy_score(train_true_cls, train_pred_cls)
        avg_per_class_acc = metrics.balanced_accuracy_score(train_true_cls, train_pred_cls)
        train_true_seg = np.concatenate(train_true_seg, axis=0)
        train_pred_seg = np.concatenate(train_pred_seg, axis=0)
        outstr = 'Train %d, loss: %.6f, train acc: %.6f, train avg acc: %.6f' % (epoch,
                                                                                                  train_loss * 1.0 / count,
                                                                                                  train_acc,
                                                                                                  avg_per_class_acc)
        io.cprint(outstr)
        writer_train_accuracy.add_scalar('Train/accuracy', train_acc, epoch)
        writer_train_balanced_accuracy.add_scalar('Train/balanced_accuracy', avg_per_class_acc, epoch)

        ####################
        # Test
        ####################
        test_loss = 0.0
        count = 0.0
        model.eval()
        test_true_cls = []
        test_pred_cls = []
        test_true_seg = []
        test_pred_seg = []

        with tqdm(test_loader, desc = "Testing epoch {}".format(epoch)) as t:
            for data, seg, _ in test_loader:
                data, seg = data.to(device), seg.to(device)
                data = data.permute(0, 2, 1).float()
                batch_size = data.size()[0]
                seg_pred = model(data)
                seg_pred = F.softmax(seg_pred, dim = 1)
                seg_pred = seg_pred.permute(0, 2, 1).contiguous()
                loss = criterion(seg_pred.view(-1, num_classes), seg.view(-1, 1).squeeze().long())
                pred = seg_pred.max(dim=2)[1]
                count += batch_size
                test_loss += loss.item() * batch_size
                seg_np = seg.cpu().numpy()
                pred_np = pred.detach().cpu().numpy()
                test_true_cls.append(seg_np.reshape(-1))
                test_pred_cls.append(pred_np.reshape(-1))
                test_true_seg.append(seg_np)
                test_pred_seg.append(pred_np)

                balanced_test_acc = metrics.balanced_accuracy_score(seg_np.reshape(-1), pred_np.reshape(-1))
                test_acc = metrics.accuracy_score(seg_np.reshape(-1), pred_np.reshape(-1))
                t.set_postfix(A = test_acc, BA = balanced_test_acc)
                t.update()
        test_true_cls = np.concatenate(test_true_cls)
        test_pred_cls = np.concatenate(test_pred_cls)
        test_acc = metrics.accuracy_score(test_true_cls, test_pred_cls)
        avg_per_class_acc = metrics.balanced_accuracy_score(test_true_cls, test_pred_cls)
        test_true_seg = np.concatenate(test_true_seg, axis=0)
        test_pred_seg = np.concatenate(test_pred_seg, axis=0)
        outstr = 'Test %d, loss: %.6f, test acc: %.6f, test avg acc: %.6f' % (epoch,
                                                                                              test_loss * 1.0 / count,
                                                                                              test_acc,
                                                                                              avg_per_class_acc)
        io.cprint(outstr)
        writer_test_accuracy.add_scalar('Test/accuracy', test_acc, epoch)
        writer_test_balanced_accuracy.add_scalar('Test/balanced_accuracy', avg_per_class_acc, epoch)
        

        # only save a model and its current training state if it's better than any iteration we've seen yet (sort of quality assurance)
        if avg_per_class_acc > best_per_class_acc:
            best_per_class_acc = avg_per_class_acc
            savepath = 'checkpoints/%s/models/%s.t7' % (exp_name, model_label)
            io.cprint('Saving the best model at %s' % savepath)
            state = {
                'epoch': epoch,
                'mBPCA': best_per_class_acc,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': opt.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
            }
            torch.save(state, savepath)

    writer_train_loss.close()
    writer_train_accuracy.close()
    writer_train_balanced_accuracy.close()
    writer_test_accuracy.close()
    writer_test_balanced_accuracy.close()

def train_args(args, io):
    if type(args.exclude_classes) == list:
        exclude_classes = [i for i in args.exclude_classes if i >= 0]
    else:
        exclude_classes = []
    train(
        args.k,
        io,
        data_dir = args.data_dir,
        num_points = args.num_points,
        block_size = args.block_size,
        epochs = args.epochs,
        num_classes = args.num_classes,
        num_features = args.num_features,
        train_batch_size = args.batch_size,
        test_batch_size = args.test_batch_size,
        min_class_num = args.min_class_num,
        use_sgd = args.use_sgd,
        lr = args.lr,
        momentum = args.momentum,
        dropout = args.dropout,
        emb_dims = args.emb_dims,
        sample_num = args.sample_num,
        scheduler = args.scheduler,
        test_prop = args.test_prop,
        use_all_points = args.use_all_points,
        cuda = args.cuda,
        model_label = args.model_label, 
        num_threads = args.num_threads, 
        exclude_classes = exclude_classes,
        num_interop_threads = args.num_interop_threads,
        model_root = args.model_root,
        exp_name = args.exp_name
    )