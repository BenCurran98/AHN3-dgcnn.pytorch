import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from data import FugroDataset, FugroDataset_eval
from model import DGCNN_semseg
import numpy as np
from torch.utils.data import DataLoader
from util import *
import sklearn.metrics as metrics
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


def test(args, io):
    DUMP_DIR = args.test_visu_dir
    all_true_cls = []
    all_pred_cls = []
    all_true_seg = []
    all_pred_seg = []
    for test_area in [1]:
        test_area = str(test_area)
        if (args.test_area == 'all') or (test_area == args.test_area):
            dataset = FugroDataset_eval(split='test', data_root=args.data_dir, num_point=args.num_points,
                                   block_size=args.block_size, use_all_points=args.use_all_points)
            test_loader = DataLoader(dataset, batch_size=args.test_batch_size, shuffle=False, drop_last=False)

            room_idx = np.array(dataset.room_idxs)
            num_blocks = len(room_idx)

            fout_data_label = []
            true_data_labels = []
            for room_id in np.unique(room_idx):
                print('room id: ', room_id)
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

            device = torch.device("cuda" if args.cuda else "cpu")

            io.cprint('Start overall evaluation...')

            # Try to load models
            if args.model == 'dgcnn':
                model = DGCNN_semseg(args).to(device)
            else:
                raise Exception("Not implemented")

            # model = nn.DataParallel(model)
            if args.cuda:
                checkpoint = torch.load(os.path.join(args.model_root, 'model_%s.t7' % args.test_area))
            else:
                checkpoint = torch.load(os.path.join(args.model_root, 'model_%s.t7' % args.test_area), map_location=torch.device('cpu'))
            
            print("model: ", os.path.join(args.model_root, 'model_%s.t7' % args.test_area))
            model.load_state_dict(checkpoint['model_state_dict'])
            model = model.eval()

            count_parameters(model)

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

                # write prediction results
                for batch_id in range(batch_size):
                    pts = data[batch_id, :, :]
                    # print("Points: ", pts.shape)
                    
                    pts = pts.permute(1, 0).float()
                    l = seg[batch_id, :]
                    pred_ = pred[batch_id, :]
                    logits = seg_pred[batch_id, :, :]
                    # compute room_id
                    room_id = room_idx[num_batch + batch_id]
                    for i in range(pts.shape[0]):
                        fout_data_label[room_id].write('%f %f %f %d %d %f %f %f %f %f\n' % (
                            pts[i, 0]*dataset.room_coord_max[room_id][0], pts[i, 1]*dataset.room_coord_max[room_id][1], pts[i, 2]*dataset.room_coord_max[room_id][2],
                            pred_[i], l[i], logits[i, 0], logits[i, 1], logits[i, 2], logits[i, 3], logits[i, 4]))  # xyzRGB pred gt
                        true_data_labels[room_id].write("%d\n" % l[i])
                num_batch += batch_size
                torch.cuda.empty_cache()

            for room_id in np.unique(room_idx):
                fout_data_label[room_id].close()

            test_ious = calculate_sem_IoU(test_pred_cls, test_true_cls, args.num_classes)
            test_true_cls = np.concatenate(test_true_cls)
            test_pred_cls = np.concatenate(test_pred_cls)
            test_acc = metrics.accuracy_score(test_true_cls, test_pred_cls)
            avg_per_class_acc = metrics.balanced_accuracy_score(test_true_cls, test_pred_cls)
            
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

