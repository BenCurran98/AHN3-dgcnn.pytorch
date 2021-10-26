from __future__ import print_function
import os
import torch
import gc
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from data import FugroDataset, collate_pcs
from model import DGCNN
import numpy as np
from torch.utils.data import DataLoader
from util import *
import sklearn.metrics as metrics
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

def train(k, io, 
            data_dir = "/media/ben/ExtraStorage/InnovationConference/Datasets/data_as_S3DIS_NRI_NPY",
            num_points = 7000,
            epochs = 30,
            num_classes = 5,
            num_features = 4,
            train_batch_size = 8,
            validation_batch_size = 8,
            use_sgd = False,
            lr = 0.001,
            momentum = 0.9,
            dropout = 0.5,
            emb_dims = 1024,
            sample_num = 5,
            scheduler = "cos",
            validation_prop = 0.2,
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
        data_dir (str, optional): Directory containing the dataset in NPY format. Defaults to "/media/ben/ExtraStorage/InnovationConference/Datasets/data_as_S3DIS_NRI_NPY".
        num_points (int, optional): Number of points to sample from each block. Defaults to 5000.
        block_size (float, optional): Size of blocks to sample from each tile. Defaults to 30.0.
        epochs (int, optional): Number of epochs to train on. Defaults to 30.
        num_classes (int, optional): Number of classes to train on. Defaults to 5.
        train_batch_size (int, optional): Number of training samples in each batch. Defaults to 8.
        validation_batch_size (int, optional): Number of test samples in each batch. Defaults to 8.
        use_sgd (bool, optional): Indicates whether to use Stochastic Gradient Descent as an optimiser. Defaults to False.
        lr (float, optional): Learning rate of optimiser. Defaults to 0.001.
        momentum (float, optional): Momentum of optimiser (only used for SGD). Defaults to 0.9.
        dropout (float, optional): Dropout probability for dropout layer in model. Defaults to 0.5.
        emb_dims (int, optional): Dimensions to embed the global feature space into. Defaults to 1024.
        sample_num (int, optional): Number of blocks to sample from each tile. Defaults to 5.
        scheduler (str, optional): Schedules the adjustment of the learning rate. Defaults to "cos".
        validation_prop (float, optional): Proportion of the dataset to use for testing/validation. Defaults to 0.2.
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

    train_data = FugroDataset(split='train', data_root=data_dir, 
                                num_point=num_points,
                                use_all_points = use_all_points, 
                                validation_prop = validation_prop, 
                                classes = range(num_classes))
    train_loader = DataLoader(
                                train_data, 
                                num_workers=num_threads, 
                                batch_size=train_batch_size,
                                shuffle=True, drop_last=True,
                                collate_fn = collate_pcs)

    validation_data = FugroDataset(split='validation', 
                                    data_root=data_dir, 
                                    num_point=num_points,
                                    validation_prop = validation_prop, 
                                    classes = range(num_classes))
    validation_loader = DataLoader(
                                    validation_data, 
                                    num_workers=num_threads, 
                                    batch_size=validation_batch_size,
                                    shuffle=True,
                                    drop_last=True,
                                    collate_fn = collate_pcs)

    device = torch.device("cuda" if cuda else "cpu")
    print("Using ", device)

    model = DGCNN(num_classes, num_features, k, 
                    ropout = dropout, 
                    emb_dims = emb_dims, 
                    cuda = cuda)

    print(str(model))
    print(count_parameters(model))
    
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
    writer_validation_accuracy = SummaryWriter(os.path.join(log_dir), 'validation_accuracy')
    writer_validation_balanced_accuracy = SummaryWriter(os.path.join(log_dir), 'balanced_validation_accuracy')

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
        train_true_labels = []
        train_pred_labels = []

        with tqdm(train_loader, desc = "Training epoch {}".format(epoch)) as t:
            for data, labels, idx in train_loader:
                mask = []
                for i in range(data.shape[0]):
                    this_mask = train_data.create_train_mask(idx[i].item(), 
                                                            data.shape[1], 
                                                            exclude_classes = exclude_classes)
                    mask.append(this_mask)
                mask = np.concatenate(mask, axis = 0)
                mask = torch.tensor(mask)
                data, labels, mask = data.to(device), labels.to(device), mask.to(device)
                data = data.permute(0, 2, 1).float()
                batch_size = data.size()[0]
                opt.zero_grad()
                labels_pred = model(data) # batch_size * num_points * num_classes
                labels_pred = F.softmax(labels_pred, dim = 1)

                # only use the points indicated by the mask in back propogation- this acts as a kind of label balancing
                focus_labels = num_classes * torch.ones_like(labels)
                focus_pred = torch.zeros((labels_pred.shape[0], num_classes + 1, labels_pred.shape[2]))
                for i in range(mask.shape[0]):
                    masked_idxs = np.where(mask[i, :])[0]
                    focus_labels[i, masked_idxs] = labels[i, masked_idxs]
                    focus_pred[i, :, masked_idxs] = torch.cat((labels_pred[i, :, masked_idxs], torch.zeros(1, len(masked_idxs))), dim = 0)
                    for j in range(len(mask[i, :])):
                        if mask[i, j] == 0:
                            focus_pred[i, num_classes, j] = 1
                
                labels = focus_labels
                labels_pred = focus_pred
                labels_pred = labels_pred.permute(0, 2, 1).contiguous()

                loss = criterion(labels_pred.view(-1, num_classes + 1), labels.view(-1, 1).squeeze().long())
                loss.backward()
                opt.step()
                
                pred = labels_pred.max(dim=2)[1]  # (batch_size, num_points)
                count += batch_size
                train_loss += loss.item() * batch_size
                niter += batch_size
                writer_train_loss.add_scalar('Train/loss', loss.item(), niter)
                labels_np = labels.cpu().numpy()  # (batch_size, num_points)
                pred_np = pred.detach().cpu().numpy()  # (batch_size, num_points)
                train_true_cls.append(labels_np.reshape(-1))  # (batch_size * num_points)
                train_pred_cls.append(pred_np.reshape(-1))  # (batch_size * num_points)
                train_true_labels.append(labels_np)
                train_pred_labels.append(pred_np)
                
                # Report on both overall accuracy and label balanced accuracy
                true_labels = labels_np.reshape(-1)
                true_labels_idxs = np.where(true_labels != num_classes)[0]
                true_labels = true_labels[true_labels_idxs]
                pred_labels = pred_np.reshape(-1)
                pred_idxs = np.where(pred_labels != num_classes)[0]
                pred_labels = pred_labels[pred_idxs]
                balanced_train_acc = metrics.balanced_accuracy_score(true_labels, pred_labels)
                train_acc = metrics.accuracy_score(true_labels, pred_labels)
                t.set_postfix(A = train_acc, BA = balanced_train_acc)
                t.update()

                gc.collect()

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
        train_true_labels = np.concatenate(train_true_labels, axis=0)
        train_pred_labels = np.concatenate(train_pred_labels, axis=0)
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
        validation_loss = 0.0
        count = 0.0
        model.eval()
        validation_true_cls = []
        validation_pred_cls = []
        validation_true_labels = []
        validation_pred_labels = []

        with tqdm(validation_loader, desc = "Testing epoch {}".format(epoch)) as t:
            for data, labels, _ in validation_loader:
                data, labels = data.to(device), labels.to(device)
                data = data.permute(0, 2, 1).float()
                batch_size = data.size()[0]
                labels_pred = model(data)
                labels_pred = F.softmax(labels_pred, dim = 1)
                labels_pred = labels_pred.permute(0, 2, 1).contiguous()
                loss = criterion(labels_pred.view(-1, num_classes), labels.view(-1, 1).squeeze().long())
                pred = labels_pred.max(dim=2)[1]
                count += batch_size
                validation_loss += loss.item() * batch_size
                labels_np = labels.cpu().numpy()
                pred_np = pred.detach().cpu().numpy()
                validation_true_cls.append(labels_np.reshape(-1))
                validation_pred_cls.append(pred_np.reshape(-1))
                validation_true_labels.append(labels_np)
                validation_pred_labels.append(pred_np)

                balanced_validation_acc = metrics.balanced_accuracy_score(labels_np.reshape(-1), pred_np.reshape(-1))
                validation_acc = metrics.accuracy_score(labels_np.reshape(-1), pred_np.reshape(-1))
                t.set_postfix(A = validation_acc, BA = balanced_validation_acc)
                t.update()
        validation_true_cls = np.concatenate(validation_true_cls)
        validation_pred_cls = np.concatenate(validation_pred_cls)
        validation_acc = metrics.accuracy_score(validation_true_cls, validation_pred_cls)
        avg_per_class_acc = metrics.balanced_accuracy_score(validation_true_cls, validation_pred_cls)
        validation_true_labels = np.concatenate(validation_true_labels, axis=0)
        validation_pred_labels = np.concatenate(validation_pred_labels, axis=0)
        outstr = 'Test %d, loss: %.6f, test acc: %.6f, test avg acc: %.6f' % (epoch,
                                                                                              validation_loss * 1.0 / count,
                                                                                              validation_acc,
                                                                                              avg_per_class_acc)
        io.cprint(outstr)
        writer_validation_accuracy.add_scalar('Test/accuracy', validation_acc, epoch)
        writer_validation_balanced_accuracy.add_scalar('Test/balanced_accuracy', avg_per_class_acc, epoch)
        

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
    writer_validation_accuracy.close()
    writer_validation_balanced_accuracy.close()

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
        epochs = args.epochs,
        num_classes = args.num_classes,
        num_features = args.num_features,
        train_batch_size = args.batch_size,
        validation_batch_size = args.validation_batch_size,
        use_sgd = args.use_sgd,
        lr = args.lr,
        momentum = args.momentum,
        dropout = args.dropout,
        emb_dims = args.emb_dims,
        sample_num = args.sample_num,
        scheduler = args.scheduler,
        validation_prop = args.validation_prop,
        use_all_points = args.use_all_points,
        cuda = args.cuda,
        model_label = args.model_label, 
        num_threads = args.num_threads, 
        exclude_classes = exclude_classes,
        num_interop_threads = args.num_interop_threads,
        model_root = args.model_root,
        exp_name = args.exp_name
    )