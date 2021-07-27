#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: An Tao
@Contact: ta19@mails.tsinghua.edu.cn
@File: main_semseg.py
@Time: 2020/2/24 7:17 PM
"""

from __future__ import print_function
import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from data import S3DISDataset
from data import S3DISDataset_eval
from model import DGCNN_semseg
import numpy as np
from torch.utils.data import DataLoader
from util import cal_loss, IOStream
import sklearn.metrics as metrics
# from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


def _init_():
    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')
    if not os.path.exists('checkpoints/' + args.exp_name):
        os.makedirs('checkpoints/' + args.exp_name)
    if not os.path.exists('checkpoints/' + args.exp_name + '/' + 'models'):
        os.makedirs('checkpoints/' + args.exp_name + '/' + 'models')
    os.system('cp main_semseg.py checkpoints' + '/' + args.exp_name + '/' + 'main_semseg.py.backup')
    os.system('cp model.py checkpoints' + '/' + args.exp_name + '/' + 'model.py.backup')
    os.system('cp util.py checkpoints' + '/' + args.exp_name + '/' + 'util.py.backup')
    os.system('cp data.py checkpoints' + '/' + args.exp_name + '/' + 'data.py.backup')


def calculate_sem_IoU(pred_np, seg_np, num_classes):  # num_classes: S3DIS 13
    I_all = np.zeros(num_classes)
    U_all = np.zeros(num_classes)
    for sem_idx in range(len(seg_np)):
        for sem in range(num_classes):
            I = np.sum(np.logical_and(pred_np[sem_idx] == sem, seg_np[sem_idx] == sem))
            U = np.sum(np.logical_or(pred_np[sem_idx] == sem, seg_np[sem_idx] == sem))
            I_all[sem] += I
            U_all[sem] += U
    return I_all / U_all


def train(args, io):
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    # sample_rate=1.5 to make sure some overlap
    train_loader = DataLoader(
        S3DISDataset(split='train', data_root=args.data_dir, num_point=args.num_points,
                     block_size=args.block_size,
                     sample_rate=1.5, num_class=args.num_classes), num_workers=8, batch_size=args.batch_size,
        shuffle=True, drop_last=True)

    test_loader = DataLoader(
        S3DISDataset(split='test', data_root=args.data_dir, num_point=args.num_points,
                     block_size=args.block_size,
                     sample_rate=1.5, num_class=args.num_classes), num_workers=8, batch_size=args.test_batch_size,
        shuffle=True, drop_last=True)

    device = torch.device("cuda" if args.cuda else "cpu")

    # Try to load models
    if args.model == 'dgcnn':
        model = DGCNN_semseg(args).to(device)
    else:
        raise Exception("Not implemented")
    print(str(model))

    model = nn.DataParallel(model)
    print("Let's use", torch.cuda.device_count(), "GPUs!")

    if args.use_sgd:
        print("Use SGD")
        opt = optim.SGD(model.parameters(), lr=args.lr * 100, momentum=args.momentum, weight_decay=1e-4)
    else:
        print("Use Adam")
        opt = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

    if args.scheduler == 'cos':
        scheduler = CosineAnnealingLR(opt, args.epochs, eta_min=1e-3)
    elif args.scheduler == 'step':
        scheduler = StepLR(opt, 20, 0.5, args.epochs)

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

    # writer_train_loss = SummaryWriter(os.path.join(log_dir, 'train_loss'))
    # writer_train_accuracy = SummaryWriter(os.path.join(log_dir))
    # writer_train_iou = SummaryWriter(os.path.join(log_dir))
    # writer_test_accuracy = SummaryWriter(os.path.join(log_dir))
    # writer_test_iou = SummaryWriter(os.path.join(log_dir))

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
        train_label_seg = []

        io.cprint('Start training for Epoch %d ...' % epoch)
        for data, seg in tqdm(train_loader):
            data, seg = data.to(device), seg.to(device)
            data = data.permute(0, 2, 1).float()
            batch_size = data.size()[0]
            opt.zero_grad()
            seg_pred = model(data)
            seg_pred = seg_pred.permute(0, 2, 1).contiguous()
            loss = criterion(seg_pred.view(-1, args.num_classes), seg.view(-1, 1).squeeze().long())
            loss.backward()
            opt.step()
            pred = seg_pred.max(dim=2)[1]  # (batch_size, num_points)
            count += batch_size
            train_loss += loss.item() * batch_size
            niter += batch_size
            # writer_train_loss.add_scalar('Train/loss', loss.item(), niter)
            seg_np = seg.cpu().numpy()  # (batch_size, num_points)
            pred_np = pred.detach().cpu().numpy()  # (batch_size, num_points)
            train_true_cls.append(seg_np.reshape(-1))  # (batch_size * num_points)
            train_pred_cls.append(pred_np.reshape(-1))  # (batch_size * num_points)
            train_true_seg.append(seg_np)
            train_pred_seg.append(pred_np)
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
        # writer_train_accuracy.add_scalar('Train/accuracy', train_acc, epoch)
        # writer_train_iou.add_scalar('Train/mIOU', np.mean(train_ious), epoch)

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

        io.cprint('Start evaluation for Epoch %d ...' % epoch)
        for data, seg in tqdm(test_loader):
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
        # writer_test_accuracy.add_scalar('Test/accuracy', test_acc, epoch)
        # writer_test_iou.add_scalar('Test/mIOU', np.mean(test_ious), epoch)

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

    # writer_train_loss.close()
    # writer_train_accuracy.close()
    # writer_train_iou.close()
    # writer_test_accuracy.close()
    # writer_test_iou.close()


def test(args, io):
    DUMP_DIR = args.test_visu_dir
    all_true_cls = []
    all_pred_cls = []
    all_true_seg = []
    all_pred_seg = []
    for test_area in range(1, 5):
        test_area = str(test_area)
        if (args.test_area == 'all') or (test_area == args.test_area):
            dataset = S3DISDataset_eval(split='test', data_root=args.data_dir, num_point=args.num_points, test_area=args.test_area,
                                   block_size=args.block_size, stride=args.block_size, num_class=args.num_classes, num_thre=100, use_all_points=True)
            test_loader = DataLoader(dataset, batch_size=args.test_batch_size, shuffle=False, drop_last=False)

            room_idx = np.array(dataset.room_idxs)
            num_blocks = len(room_idx)

            fout_data_label = []
            for room_id in np.unique(room_idx):
                out_data_label_filename = 'Area_%s_room_%d_pred_gt.txt' % (test_area, room_id)
                out_data_label_filename = os.path.join(DUMP_DIR, out_data_label_filename)
                fout_data_label.append(open(out_data_label_filename, 'w+'))

            device = torch.device("cuda" if args.cuda else "cpu")

            io.cprint('Start overall evaluation...')

            # Try to load models
            if args.model == 'dgcnn':
                model = DGCNN_semseg(args).to(device)
            else:
                raise Exception("Not implemented")

            model = nn.DataParallel(model)
            if args.cuda:
                checkpoint = torch.load(os.path.join(args.model_root, 'model_%s.t7' % args.test_area))
            else:
                checkpoint = torch.load(os.path.join(args.model_root, 'model_%s.t7' % args.test_area), map_location=torch.device('cpu'))
            model.load_state_dict(checkpoint['model_state_dict'])
            model = model.eval()

            io.cprint('model_%s.t7 restored.' % args.test_area)

            test_acc = 0.0
            count = 0.0
            test_true_cls = []
            test_pred_cls = []
            test_true_seg = []
            test_pred_seg = []

            io.cprint('Start testing ...')
            num_batch = 0
            for data, seg in tqdm(test_loader):
                data, seg = data.to(device), seg.to(device)
                data = data.permute(0, 2, 1).float()
                batch_size = data.size()[0]

                seg_pred = model(data)
                seg_pred = seg_pred.permute(0, 2, 1).contiguous()
                pred = seg_pred.max(dim=2)[1]
                seg_np = seg.cpu().numpy()
                pred_np = pred.detach().cpu().numpy()
                test_true_cls.append(seg_np.reshape(-1))
                test_pred_cls.append(pred_np.reshape(-1))
                test_true_seg.append(seg_np)
                test_pred_seg.append(pred_np)
                print(np.array(seg_np).shape)

                # write prediction results
                for batch_id in range(batch_size):
                    pts = data[batch_id, :, :]
                    pts = pts.permute(1, 0).float()
                    l = seg[batch_id, :]
                    pts[:, 3:6] *= 255.0
                    pred_ = pred[batch_id, :]
                    logits = seg_pred[batch_id, :, :]
                    # compute room_id
                    room_id = room_idx[num_batch + batch_id]
                    for i in range(pts.shape[0]):
                        fout_data_label[room_id].write('%f %f %f %d %d %d %d %d %f %f %f %f\n' % (
                            pts[i, 6]*dataset.room_coord_max[room_id][0], pts[i, 7]*dataset.room_coord_max[room_id][1], pts[i, 8]*dataset.room_coord_max[room_id][2],
                            pts[i, 3], pts[i, 4], pts[i, 5], pred_[i], l[i], logits[i, 0], logits[i, 1], logits[i, 2], logits[i, 3]))  # xyzRGB pred gt
                num_batch += batch_size
                torch.cuda.empty_cache()

            for room_id in np.unique(room_idx):
                fout_data_label[room_id].close()

            test_ious = calculate_sem_IoU(test_pred_cls, test_true_cls, args.num_classes)
            test_true_cls = np.concatenate(test_true_cls)
            test_pred_cls = np.concatenate(test_pred_cls)
            test_acc = metrics.accuracy_score(test_true_cls, test_pred_cls)
            avg_per_class_acc = metrics.balanced_accuracy_score(test_true_cls, test_pred_cls)
            # test_pred_seg = np.concatenate(test_pred_seg, axis=0)
            outstr = 'Test :: test area: %s, test acc: %.6f, test avg acc: %.6f, test iou: %.6f' % (test_area,
                                                                                                    test_acc,
                                                                                                    avg_per_class_acc,
                                                                                                    np.mean(test_ious))
            io.cprint(outstr)

            # calculate confusion matrix
            conf_mat = metrics.confusion_matrix(test_true_cls, test_pred_cls)
            io.cprint('Confusion matrix:')
            io.cprint(str(conf_mat))

            all_true_cls.append(test_true_cls)
            all_pred_cls.append(test_pred_cls)
            all_true_seg.append(test_true_seg)
            all_pred_seg.append(test_pred_seg)

    if args.test_area == 'all':
        all_true_cls = np.concatenate(all_true_cls)
        all_pred_cls = np.concatenate(all_pred_cls)
        all_acc = metrics.accuracy_score(all_true_cls, all_pred_cls)
        avg_per_class_acc = metrics.balanced_accuracy_score(all_true_cls, all_pred_cls)
        all_true_seg = np.concatenate(all_true_seg, axis=0)
        all_pred_seg = np.concatenate(all_pred_seg, axis=0)
        all_ious = calculate_sem_IoU(all_pred_seg, all_true_seg, args.num_classes)
        outstr = 'Overall Test :: test acc: %.6f, test avg acc: %.6f, test iou: %.6f' % (all_acc,
                                                                                         avg_per_class_acc,
                                                                                         np.mean(all_ious))
        io.cprint(outstr)


if __name__ == "__main__":
    # Training settings
    parser = argparse.ArgumentParser(description='Point Cloud Semantic Segmentation')
    parser.add_argument('--data_dir', type=str, default='/home/ubuntu/Datasets/powercor_as_S3DIS_NRI_NPY',
                        help='Directory of data')
    parser.add_argument('--tb_dir', type=str, default='log_tensorboard',
                        help='Directory of tensorboard logs')
    parser.add_argument('--exp_name', type=str, default='powercor_integration_50epochs_p100', metavar='N',
                        help='Name of the experiment')
    parser.add_argument('--model', type=str, default='dgcnn', metavar='N',
                        choices=['dgcnn'],
                        help='Model to use, [dgcnn]')
    parser.add_argument('--dataset', type=str, default='S3DIS', metavar='N',
                        choices=['S3DIS'])
    parser.add_argument('--block_size', type=float, default=30.0,
                        help='size of one block')
    parser.add_argument('--num_classes', type=int, default=4,
                        help='number of classes in the dataset')
    parser.add_argument('--test_area', type=str, default='4', metavar='N',
                        choices=['1', '2', '3', '4', 'all'])
    parser.add_argument('--batch_size', type=int, default=12, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--test_batch_size', type=int, default=12, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--epochs', type=int, default=50, metavar='N',
                        help='number of episode to train ')
    parser.add_argument('--use_sgd', type=bool, default=False,
                        help='Use SGD')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001, 0.1 if using sgd)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--scheduler', type=str, default='cos', metavar='N',
                        choices=['cos', 'step'],
                        help='Scheduler to use, [cos, step]')
    parser.add_argument('--no_cuda', type=bool, default=False,
                        help='enables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--eval', type=bool, default=False,
                        help='evaluate the model')
    parser.add_argument('--num_points', type=int, default=4096,
                        help='num of points to use')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='dropout rate')
    parser.add_argument('--emb_dims', type=int, default=1024, metavar='N',
                        help='Dimension of embeddings')
    parser.add_argument('--k', type=int, default=20, metavar='N',
                        help='Num of nearest neighbors to use')
    parser.add_argument('--model_root', type=str, default='checkpoints/RGB_30m/models', metavar='N',
                        help='Pretrained model root')
    parser.add_argument('--test_visu_dir', default='predict',
                        help='Directory of test visualization files.')
    args = parser.parse_args()

    _init_()

    io = IOStream('checkpoints/' + args.exp_name + '/run.log')
    io.cprint(str(args))

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    if args.cuda:
        io.cprint(
            'Using GPU : ' + str(torch.cuda.current_device()) + ' from ' + str(torch.cuda.device_count()) + ' devices')
        torch.cuda.manual_seed(args.seed)
    else:
        io.cprint('Using CPU')

    if not args.eval:
        train(args, io)
    else:
        test(args, io)
