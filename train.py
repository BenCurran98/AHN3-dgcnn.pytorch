from __future__ import print_function
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from data import FugroDataset
from data import S3DISDataset_eval
from model import DGCNN_semseg
import numpy as np
from torch.utils.data import DataLoader
from util import *
import sklearn.metrics as metrics
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

def train(args, io):
    """Train a DGCNN Model

    Args:
        args (dict): Dictionary mapping keyword args to training parameters (see `__main__` for more details)
        io (IOStream): IO Channel to output logs to

    Raises:
        Exception: Not Implemented Exception for invalid model type
    """
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    torch.set_num_threads(args.num_threads)
    torch.set_num_interop_threads(args.num_interop_threads)

    train_loader = DataLoader(
        FugroDataset(split='train', data_root=args.data_dir, num_point=args.num_points,
                     block_size=args.block_size, use_all_points = args.use_all_points, test_prop = args.test_prop, sample_num = args.sample_num), num_workers=8, batch_size=args.batch_size,
        shuffle=True, drop_last=True)

    test_loader = DataLoader(
        FugroDataset(split='test', data_root=args.data_dir, num_point=args.num_points,
                     block_size=args.block_size, test_prop = args.test_prop), num_workers=8, batch_size=args.test_batch_size,
        shuffle=True, drop_last=True)

    device = torch.device("cuda" if args.cuda else "cpu")
    print("Using ", device)

    # Try to load models
    if args.model == 'dgcnn':
        model = DGCNN_semseg(args).to(device)
    else:
        raise Exception("Not implemented")

    print(str(model))
    
    if args.cuda:
        print("Let's use", torch.cuda.device_count(), "GPUs!")

    if args.use_sgd:
        print("Use SGD")
        opt = optim.SGD(model.parameters(), lr=args.lr * 100, momentum=args.momentum, weight_decay=1e-4)
    else:
        print("Use Adam")
        opt = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

    # cosine annealing is a good way to avoid gradient decay
    if args.scheduler == 'cos':
        scheduler = CosineAnnealingLR(opt, args.epochs, eta_min=1e-3)
    elif args.scheduler == 'step':
        scheduler = StepLR(opt, 20, 0.5, args.epochs)

    # check if we have a model saved somewhere- if so, load it and train
    try:
        checkpoint = torch.load(os.path.join(args.model_root, 'model_%s.t7' % args.test_area))
        start_epoch = checkpoint['epoch'] + 1
        model.load_state_dict(checkpoint['model_state_dict'])
        opt.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        best_test_iou = checkpoint['mIOU']
        io.cprint('Use pretrained model')
    except:
        io.cprint('No existing model, starting training from scratch...')
        start_epoch = 0
        best_test_iou = 0

    criterion = cal_loss

    log_dir = os.path.join(BASE_DIR, args.tb_dir)
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)

    writer_train_loss = SummaryWriter(os.path.join(log_dir, 'train_loss'))
    writer_train_accuracy = SummaryWriter(os.path.join(log_dir), 'train_accuracy')
    writer_train_iou = SummaryWriter(os.path.join(log_dir), 'train_iou')
    writer_test_accuracy = SummaryWriter(os.path.join(log_dir), 'test_accuracy')
    writer_test_iou = SummaryWriter(os.path.join(log_dir), 'test_io')

    for epoch in range(start_epoch, args.epochs):
        ####################
        # Train
        ####################
        train_loss = 0.0
        count = 0.0
        niter = epoch * len(train_loader) * args.batch_size
        model.train()
        train_true_cls = []
        train_pred_cls = []
        train_true_seg = []
        train_pred_seg = []

        with tqdm(train_loader, desc = "Training epoch {}".format(epoch)) as t:
            for data, seg, mask in train_loader:
                data, seg, mask = data.to(device), seg.to(device), mask.to(device)
                data = data.permute(0, 2, 1).float()
                batch_size = data.size()[0]
                opt.zero_grad()
                seg_pred = model(data) # batch_size * num_points * num_classes

                # only use the points indicated by the mask in back propogation- this acts as a kind of label balancing
                focus_seg = torch.zeros_like(seg)
                focus_pred = torch.zeros_like(seg_pred)
                for i in range(mask.shape[0]):
                    masked_idxs = np.where(mask[i, :])[0]
                    focus_seg[i, masked_idxs] = seg[i, masked_idxs]
                    focus_pred[i, :, masked_idxs] = seg_pred[i, :, masked_idxs]
                
                seg = focus_seg
                seg_pred = focus_pred
                seg_pred = seg_pred.permute(0, 2, 1).contiguous()

                loss = criterion(seg_pred.view(-1, args.num_classes), seg.view(-1, 1).squeeze().long())
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
                balanced_train_acc = metrics.balanced_accuracy_score(seg_np.reshape(-1), pred_np.reshape(-1))
                train_acc = metrics.accuracy_score(seg_np.reshape(-1), pred_np.reshape(-1))
                t.set_postfix(accuracy = train_acc, balanced_accuracy = balanced_train_acc)
                t.update()

        if args.scheduler == 'cos':
            scheduler.step()
        elif args.scheduler == 'step':
            if opt.param_groups[0]['lr'] > 1e-5:
                scheduler.step()
            if opt.param_groups[0]['lr'] < 1e-5:
                for param_group in opt.param_groups:
                    param_group['lr'] = 1e-5

        train_true_cls = np.concatenate(train_true_cls)
        train_pred_cls = np.concatenate(train_pred_cls)
        train_acc = metrics.accuracy_score(train_true_cls, train_pred_cls)
        avg_per_class_acc = metrics.balanced_accuracy_score(train_true_cls, train_pred_cls)
        train_true_seg = np.concatenate(train_true_seg, axis=0)
        train_pred_seg = np.concatenate(train_pred_seg, axis=0)
        train_ious = calculate_sem_IoU(train_pred_seg, train_true_seg, args.num_classes)
        outstr = 'Train %d, loss: %.6f, train acc: %.6f, train avg acc: %.6f, train iou: %.6f' % (epoch,
                                                                                                  train_loss * 1.0 / count,
                                                                                                  train_acc,
                                                                                                  avg_per_class_acc,
                                                                                                  np.mean(train_ious))
        io.cprint(outstr)
        writer_train_accuracy.add_scalar('Train/accuracy', train_acc, epoch)
        writer_train_iou.add_scalar('Train/mIOU', np.mean(train_ious), epoch)

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
                seg_pred = seg_pred.permute(0, 2, 1).contiguous()
                loss = criterion(seg_pred.view(-1, args.num_classes), seg.view(-1, 1).squeeze().long())
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
                t.set_postfix(accuracy = test_acc, balanced_accuracy = balanced_test_acc)
                t.update()
        test_true_cls = np.concatenate(test_true_cls)
        test_pred_cls = np.concatenate(test_pred_cls)
        test_acc = metrics.accuracy_score(test_true_cls, test_pred_cls)
        avg_per_class_acc = metrics.balanced_accuracy_score(test_true_cls, test_pred_cls)
        test_true_seg = np.concatenate(test_true_seg, axis=0)
        test_pred_seg = np.concatenate(test_pred_seg, axis=0)
        test_ious = calculate_sem_IoU(test_pred_seg, test_true_seg, args.num_classes)
        outstr = 'Test %d, loss: %.6f, test acc: %.6f, test avg acc: %.6f, test iou: %.6f' % (epoch,
                                                                                              test_loss * 1.0 / count,
                                                                                              test_acc,
                                                                                              avg_per_class_acc,
                                                                                              np.mean(test_ious))
        io.cprint(outstr)
        writer_test_accuracy.add_scalar('Test/accuracy', test_acc, epoch)
        writer_test_iou.add_scalar('Test/mIOU', np.mean(test_ious), epoch)

        # only save a model and its current training state if it's better than any iteration we've seen yet (sort of quality assurance)
        if np.mean(test_ious) >= best_test_iou:
            best_test_iou = np.mean(test_ious)
            savepath = 'checkpoints/%s/models/model_%s.t7' % (args.exp_name, args.test_area)
            io.cprint('Saving the best model at %s' % savepath)
            state = {
                'epoch': epoch,
                'mIOU': best_test_iou,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': opt.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
            }
            torch.save(state, savepath)

    writer_train_loss.close()
    writer_train_accuracy.close()
    writer_train_iou.close()
    writer_test_accuracy.close()
    writer_test_iou.close()