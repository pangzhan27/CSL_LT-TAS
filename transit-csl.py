import os
import argparse
import random
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import numpy as np
import copy
import logging
from copy import deepcopy
from scipy.ndimage import gaussian_filter1d

# os.environ["CUDA_VISIBLE_DEVICES"]='1'
################## Helper function
def get_frame_num_per_cls(list_file, gt_path, actions_dict):
    file_ptr = open(list_file, 'r')
    list_of_examples = file_ptr.read().split('\n')[:-1]
    file_ptr.close()

    num_per_cls = np.zeros(len(actions_dict))
    for vid in list_of_examples:
        file_ptr = open(gt_path + vid, 'r')
        contents = file_ptr.read().split('\n')[:-1]
        contents = contents[::sample_rate]
        for c in contents:
            num_per_cls[actions_dict[c]] += 1
    return num_per_cls

def get_transition_prior(list_file, gt_path, actions_dict):
    file_ptr = open(list_file, 'r')
    list_of_examples = file_ptr.read().split('\n')[:-1]
    file_ptr.close()

    trans_prior = np.zeros((len(actions_dict)+1, len(actions_dict))) # num_class + 'start', num_class
    for vid in list_of_examples:
        file_ptr = open(gt_path + vid, 'r')
        contents = file_ptr.read().split('\n')[:-1]
        contents = contents[::sample_rate]
        l, s, e = get_labels_start_end_time(contents)
        for i in range(len(l)):
            if i == 0:
                prev = len(actions_dict)
            else:
                prev = actions_dict[l[i-1]]
            cur = actions_dict[l[i]]
            trans_prior[prev, cur] += (e[i] - s[i])

    trans_prior = trans_prior/np.sum(trans_prior)
    return trans_prior


class Logger(object):
    level_relations = {
        'debug': logging.DEBUG,
        'info': logging.INFO,
        'warning': logging.WARNING,
        'error': logging.ERROR,
        'crit': logging.CRITICAL
    }

    def __init__(self, logpath, logfile, level='info',
                 fmt='%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s'):
        self.logger = logging.getLogger()
        self.logger.setLevel(self.level_relations.get(level))
        format_str = logging.Formatter(fmt)
        if not os.path.exists(logpath):
            os.makedirs(logpath)
        fh = logging.FileHandler(logfile, mode='w')
        fh.setLevel(logging.INFO)
        fh.setFormatter(format_str)
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(format_str)
        self.logger.addHandler(fh)
        self.logger.addHandler(ch)


################## Metrics
def get_labels_start_end_time(frame_wise_labels, bg_class=["background"]):
    labels = []
    starts = []
    ends = []
    last_label = frame_wise_labels[0]
    if frame_wise_labels[0] not in bg_class:
        labels.append(frame_wise_labels[0])
        starts.append(0)
    for i in range(len(frame_wise_labels)):
        if frame_wise_labels[i] != last_label:
            if frame_wise_labels[i] not in bg_class:
                labels.append(frame_wise_labels[i])
                starts.append(i)
            if last_label not in bg_class:
                ends.append(i)
            last_label = frame_wise_labels[i]
    if last_label not in bg_class:
        ends.append(i+1)
    return labels, starts, ends


def f_score(recognized, ground_truth, overlap, bg_class=["background"]):
    p_label, p_start, p_end = get_labels_start_end_time(recognized, bg_class)
    y_label, y_start, y_end = get_labels_start_end_time(ground_truth, bg_class)

    tp = 0
    fp = 0

    hits = np.zeros(len(y_label))

    for j in range(len(p_label)):
        intersection = np.minimum(p_end[j], y_end) - np.maximum(p_start[j], y_start)
        union = np.maximum(p_end[j], y_end) - np.minimum(p_start[j], y_start)
        IoU = (1.0 * intersection / union) * ([p_label[j] == y_label[x] for x in range(len(y_label))])
        # Get the best scoring segment
        idx = np.array(IoU).argmax()

        if IoU[idx] >= overlap and not hits[idx]:
            tp += 1
            hits[idx] = 1
        else:
            fp += 1
    fn = len(y_label) - sum(hits)
    return float(tp), float(fp), float(fn)


def overlap_f1(P, Y, overlap=.1, bg_class=["background"]):
    TP, FP, FN = 0, 0, 0
    for i in range(len(P)):
        tp, fp, fn = f_score(P[i], Y[i], overlap, bg_class)
        TP += tp
        FP += fp
        FN += fn
    precision = TP / float(TP + FP + 1e-8)
    recall = TP / float(TP + FN + 1e-8)
    F1 = 2 * (precision * recall) / (precision + recall + 1e-16)
    F1 = np.nan_to_num(F1)
    return F1 * 100


def accuracy(P, Y):
    total = 0.
    correct = 0
    for i in range(len(P)):
        total += len(Y[i])
        correct += (P[i] == Y[i]).sum()
    return torch.Tensor([100 * correct / total])


def levenstein_(p, y, norm=False):
    m_row = len(p)
    n_col = len(y)
    D = np.zeros([m_row + 1, n_col + 1], 'float')
    for i in range(m_row + 1):
        D[i, 0] = i
    for i in range(n_col + 1):
        D[0, i] = i

    for j in range(1, n_col + 1):
        for i in range(1, m_row + 1):
            if y[j - 1] == p[i - 1]:
                D[i, j] = D[i - 1, j - 1]
            else:
                D[i, j] = min(D[i - 1, j] + 1,
                              D[i, j - 1] + 1,
                              D[i - 1, j - 1] + 1)

    if norm:
        score = (1 - D[-1, -1] / max(m_row, n_col)) * 100
    else:
        score = D[-1, -1]

    return score


def edit_score(P, Y, norm=True, bg_class=["background"]):
    if type(P) == list:
        tmp = [edit_score(P[i], Y[i], norm, bg_class) for i in range(len(P))]
        return np.mean(tmp)
    else:
        P_, _, _ = get_labels_start_end_time(P, bg_class)
        Y_, _, _ = get_labels_start_end_time(Y, bg_class)
        return levenstein_(P_, Y_, norm)


# balanced metric
def f_score_ana(recognized, ground_truth, overlap, actions_dict, bg_class=["background"]):
    p_label, p_start, p_end = get_labels_start_end_time(recognized, bg_class)
    y_label, y_start, y_end = get_labels_start_end_time(ground_truth, bg_class)

    tp = np.zeros(len(actions_dict))
    fp = np.zeros(len(actions_dict))
    fn = np.zeros(len(actions_dict))
    hits = np.zeros(len(y_label))

    for j in range(len(p_label)):
        intersection = np.minimum(p_end[j], y_end) - np.maximum(p_start[j], y_start)
        union = np.maximum(p_end[j], y_end) - np.minimum(p_start[j], y_start)
        IoU = (1.0 * intersection / union) * ([p_label[j] == y_label[x] for x in range(len(y_label))])
        # Get the best scoring segment
        idx = np.array(IoU).argmax()

        if IoU[idx] >= overlap and not hits[idx]:
            tp[actions_dict[p_label[j]]] += 1
            hits[idx] = 1
        else:
            fp[actions_dict[p_label[j]]] += 1
    for j in range(len(y_label)):
        if hits[j] == 0:
            fn[actions_dict[y_label[j]]] += 1

    return tp, fp, fn


def overlap_f1_macro(P, Y, overlap=.1, bg_class=["background"]):
    TP, FP, FN = 0, 0, 0
    for i in range(len(P)):
        tp, fp, fn = f_score_ana(P[i], Y[i], overlap, actions_dict, bg_class)
        TP += tp
        FP += fp
        FN += fn
    precision = TP / (TP + FP + 1e-8)
    recall = TP / (TP + FN + 1e-8)
    F1 = 2 * (precision * recall) / (precision + recall + 1e-16)
    F1 = np.nan_to_num(F1)
    return F1 * 100


def b_accuracy(P, Y):
    total = np.zeros(len(actions_dict))
    correct = np.zeros(len(actions_dict))
    cover = np.zeros(len(actions_dict))
    for i in range(len(P)):
        num = min(len(P[i]), len(Y[i]))
        for j in range(num):
            if P[i][j] == Y[i][j]:
                correct[actions_dict[Y[i][j]]] += 1
            total[actions_dict[Y[i][j]]] += 1
            cover[actions_dict[P[i][j]]] += 1

    avg_acc = 100 * correct / (total + 1e-8)
    avg_prec = 100 * correct / (cover + 1e-8)
    return avg_acc, avg_prec


################## Dataloader
class BatchGenerator(torch.utils.data.Dataset):
    def __init__(self, num_classes, actions_dict, gt_path, features_path, sample_rate, vid_list_file):
        self.list_of_examples = list()
        self.index = 0
        self.num_classes = num_classes
        self.actions_dict = actions_dict
        self.gt_path = gt_path
        self.features_path = features_path
        self.sample_rate = sample_rate
        self.read_data(vid_list_file)

    def __len__(self):
        return len(self.list_of_examples)

    def read_data(self, vid_list_file):
        file_ptr = open(vid_list_file, 'r')
        self.list_of_examples = file_ptr.read().split('\n')[:-1]
        file_ptr.close()
        random.shuffle(self.list_of_examples)

    def getitem(self, index):
        vid = self.list_of_examples[index]

        features = np.load(self.features_path + vid.split('.')[0] + '.npy')
        file_ptr = open(self.gt_path + vid, 'r')
        content = file_ptr.read().split('\n')[:-1]
        classes = np.zeros(min(np.shape(features)[1], len(content)))
        for i in range(len(classes)):
            classes[i] = self.actions_dict[content[i]]
        input = features[:, ::self.sample_rate]
        target = classes[::self.sample_rate]

        batch_input_tensor = torch.from_numpy(input).float()
        batch_target_tensor = torch.from_numpy(target).long()
        mask = torch.ones(self.num_classes, np.shape(input)[1])

        return batch_input_tensor, batch_target_tensor, mask, vid

    def __getitem__(self, index):
        return self.getitem(index)


################## Model (change loss)
class MultiStageModel(nn.Module):
    def __init__(self, num_stages, num_layers, num_f_maps, dim, num_classes):
        super(MultiStageModel, self).__init__()
        self.stage1 = SingleStageModel(num_layers, num_f_maps, dim, num_classes)
        self.stages = nn.ModuleList(
            [copy.deepcopy(SingleStageModel(num_layers, num_f_maps, num_classes, num_classes)) for s in
             range(num_stages - 1)])

    def forward(self, x, mask):
        out, feat = self.stage1(x, mask)
        outputs = out.unsqueeze(0)
        for i, s in enumerate(self.stages):
            out, feat = s(F.softmax(out, dim=1) * mask[:, 0:1, :], mask)
            outputs = torch.cat((outputs, out.unsqueeze(0)), dim=0)
        return outputs, feat

class SingleStageModel(nn.Module):
    def __init__(self, num_layers, num_f_maps, dim, num_classes):
        super(SingleStageModel, self).__init__()
        self.conv_1x1 = nn.Conv1d(dim, num_f_maps, 1)
        self.layers = nn.ModuleList(
            [copy.deepcopy(DilatedResidualLayer(2 ** i, num_f_maps, num_f_maps)) for i in range(num_layers)])
        self.conv_out = nn.Conv1d(num_f_maps, num_classes, 1)

    def forward(self, x, mask):
        out = self.conv_1x1(x)
        for layer in self.layers:
            out = layer(out, mask)
        out1 = self.conv_out(out) * mask[:, 0:1, :]
        return out1, out


class DilatedResidualLayer(nn.Module):
    def __init__(self, dilation, in_channels, out_channels):
        super(DilatedResidualLayer, self).__init__()
        self.conv_dilated = nn.Conv1d(in_channels, out_channels, 3, padding=dilation, dilation=dilation)
        self.conv_1x1 = nn.Conv1d(out_channels, out_channels, 1)
        self.dropout = nn.Dropout()

    def forward(self, x, mask):
        out = F.relu(self.conv_dilated(x))
        out = self.conv_1x1(out)
        out = self.dropout(out)
        return (x + out) * mask[:, 0:1, :]


class TransitionSat(torch.nn.Module):
    def __init__(self, prior, device='cpu'):
        super(TransitionSat, self).__init__()
        self.device = device
        self.lambdas = np.zeros((num_classes + 1, num_classes))
        self.val_lr = args.lr
        self.num_classes = num_classes
        dummy_mat = np.eye(num_classes)
        for i in range(self.num_classes):
            dummy_mat[i, i] = 1.0 / class_prior[i]
        # + 1 dimension for "start"
        self.dummy_mat = np.repeat(dummy_mat[:, :, np.newaxis], num_classes+1, axis=2)
        self.prior = prior
        self.G = self.dummy_mat
        self.gain(self.G)
        self.valid_last, self.valid_cur = np.where(trans_prior>0)


    def gain(self, G):
        self.adjustment = []
        for i in range(G.shape[-1]):
            D = np.diag(np.diag(G[:,:,i]))** args.tau
            D = torch.tensor(np.diag(D), requires_grad=False)
            norm_D = D * len(D)/torch.sum(D)
            self.adjustment.append(norm_D.to(self.device).unsqueeze(0))

        self.adjustment = torch.cat(self.adjustment, dim=0).to(self.device)

    def update(self, CM):
        all_trans_rec = []
        for i in range(len(self.valid_last)):
            all_trans_rec.append(CM[self.valid_cur[i], self.valid_cur[i], self.valid_last[i]]/trans_prior[self.valid_last[i], self.valid_cur[i]])
            assert np.abs(trans_prior[self.valid_last[i], self.valid_cur[i]] - np.sum(CM[self.valid_cur[i], :, self.valid_last[i]])) < 1e-8
            #print(trans_prior[self.valid_last[i], self.valid_cur[i]],np.sum(CM[self.valid_cur[i], :, self.valid_last[i]]))

        avg_trans_rec = np.mean(all_trans_rec)
        for i in range(len(self.valid_last)):
            new_lambda = self.lambdas[self.valid_last[i], self.valid_cur[i]] - \
                         self.val_lr * (CM[self.valid_cur[i], self.valid_cur[i], self.valid_last[i]] / trans_prior[self.valid_last[i], self.valid_cur[i]] -
                                        args.clip*avg_trans_rec ) * (trans_prior[self.valid_last[i], self.valid_cur[i]] / class_prior[self.valid_cur[i]])
            new_lambda = min(max(0, new_lambda), 1)
            self.lambdas[self.valid_last[i], self.valid_cur[i]] = new_lambda

        G = self.dummy_mat
        for k in range(self.num_classes+1): # for each last step
            for j in range(self.num_classes):
                G[j, j, k] += self.lambdas[k,j] / class_prior[j]
        self.gain(G)
        return

    def forward(self, inputs, targets,  prev, reduction='mean'):
        new_weight = self.adjustment[prev]
        log_probs = F.log_softmax(inputs, dim=1)
        one_hot = F.one_hot(targets, num_classes=num_classes)
        product = one_hot * log_probs*new_weight
        if reduction == 'mean':
            return -1 * torch.mean(torch.sum(product, 1))
        else:
            return -1 * torch.sum(product, 1)


def find_previous_seg(gt):
    previous = np.zeros(len(gt))
    # if use \hat{y}_{t-1}, gt is useful in this case
    gtrue = gt.cpu().numpy()
    gl, gs, ge = get_labels_start_end_time(gtrue)
    all_points = gs + [len(gtrue)]

    for i in range(len(all_points)-1):
        # for the first segment, its previous seg is num_class, namely an extra class to denote "start"
        if i ==0:
            previous[all_points[i]: all_points[i+1]] = num_classes
        else:
            labels = gtrue[all_points[i-1]: all_points[i]]
            previous[all_points[i]: all_points[i+1]] = labels[0]
            assert np.min(labels) == np.max(labels)

    return torch.from_numpy(previous).long().to(device)


class KNNClassifier(nn.Module):
    def __init__(self, feat_dim=512, num_classes=48, feat_type='l2n'):
        super(KNNClassifier, self).__init__()
        assert feat_type in ['un', 'l2n', 'cl2n'], "feat_type is wrong!!!"
        self.feat_dim = feat_dim
        self.num_classes = num_classes
        self.feat_mean = torch.randn(feat_dim)
        self.centroids = torch.randn(num_classes, feat_dim)
        self.feat_type = feat_type
        self.initialized = False

    def update(self, cfeats):
        mean = cfeats['mean']
        centroids = cfeats['{}cs'.format(self.feat_type)]
        label_map = cfeats['label_map']

        mean = torch.from_numpy(mean)
        centroids = torch.from_numpy(centroids)
        self.feat_mean.copy_(mean)
        self.centroids = centroids
        self.label_map = torch.from_numpy(label_map).to(device)
        if torch.cuda.is_available():
            self.feat_mean = self.feat_mean.cuda()
            self.centroids = self.centroids.cuda()
        self.initialized = True

    def forward(self, inputs):
        centroids = self.centroids
        feat_mean = self.feat_mean

        # Feature transforms
        if self.feat_type == 'cl2n':
            inputs = inputs - feat_mean

        if self.feat_type in ['l2n', 'cl2n']:
            norm_x = torch.norm(inputs, 2, 1, keepdim=True)
            inputs = inputs / norm_x

        # Logit calculation
        logits = self.l2_similarity(inputs, centroids)

        return logits

    def l2_similarity(self, A, B):
        # input A: [bs, fd] (batch_size x feat_dim)
        # input B: [nC, fd] (num_classes x feat_dim)
        feat_dim = A.size(1)

        AB = torch.mm(A, B.t())
        AA = (A ** 2).sum(dim=1, keepdim=True)
        BB = (B ** 2).sum(dim=1, keepdim=True)
        dist = AA + BB.t() - 2 * AB

        return -dist

def postprocess(preds, ncm_preds, logits):
    prob = F.softmax(logits, dim=-1)
    out_pred = deepcopy(preds)
    #decide the segment
    transition_pred = np.where((preds[:-1] - preds[1:]) != 0)[0] + 1
    idx = np.concatenate([[0], transition_pred, [len(preds)]])
    for j in range(len(idx) - 1):
        count = np.bincount(ncm_preds[idx[j]: idx[j + 1]])
        modes = np.where(count == count.max())[0]
        if len(modes) == 1:
            mode = modes[0]
        elif len(modes) == 0:
            print(preds)
            print(idx)
            print(count)
        else:
            prob_sum_max = 0
            for m in modes:
                prob_sum = prob[idx[j]: idx[j + 1], m].sum()
                if prob_sum_max < prob_sum:
                    mode = m
                    prob_sum_max = prob_sum
        out_pred[idx[j]: idx[j + 1]] = mode
    return out_pred

################## Trainer (change loss)
class Trainer:
    def __init__(self, model, log, sample_rate, **kwargs):
        set_seed(seed)
        self.model = model(**kwargs)
        self.ce = nn.CrossEntropyLoss(ignore_index=-100)
        self.ce_split = nn.CrossEntropyLoss(ignore_index=-100, reduction='none')
        self.mse = nn.MSELoss(reduction='none')
        self.hyb_loss = TransitionSat(class_prior, device)
        self.boundary_criterion = nn.BCELoss(reduction='none')
        self.num_classes = kwargs.get('num_classes', 0)
        self.sample_rate = sample_rate
        assert self.num_classes > 0, "wrong class numbers"
        self.log = log
        self.log.logger.info('Model Size: {}'.format(sum(p.numel() for p in self.model.parameters())))
        self.knn = KNNClassifier(64, self.num_classes)

    def train(self, save_dir, num_epochs, batch_size, learning_rate, device, actions_dict,
              batch_gen_tst=None):

        self.model.train()
        self.model.to(device)
        resume_epoch = 0
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        best_score, best_score_ncm = -10000, -10000
        if args.resume > 0:
            state = torch.load("./models/" + args.dataset + "/split_" + args.split + TYPE + "/epoch-last" + ".model",
                               map_location=device)
            resume_epoch = state['epoch'] + 1
            self.model.load_state_dict(state['net'])
            best_score, best_score_ncm = state['score'], state['score_ncm']

        all_lambda, all_cm, all_epoch, all_adjust = [], [], [], []
        for epoch in range(resume_epoch, num_epochs):
            epoch_loss = 0
            correct = 0
            total = 0
            ce_loss, smooth_loss, boundary_loss = [0, 0, 0, 0], [0, 0, 0, 0], 0
            for _, items in enumerate(train_loader):
                batch_input, batch_target, mask, vids = items
                batch_input, batch_target, mask = batch_input.to(device), batch_target.to(device),  mask.to(device)

                optimizer.zero_grad()

                ps, _ = self.model(batch_input, mask)
                loss = 0
                for i, p in enumerate(ps):
                    if i == len(ps) - 1:
                        previous = find_previous_seg(batch_target.view(-1))
                        s_ce_loss = self.hyb_loss(p.transpose(2, 1).contiguous().view(-1, self.num_classes), batch_target.squeeze(0), previous)

                    else:
                        s_ce_loss = self.ce(p.transpose(2, 1).contiguous().view(-1, self.num_classes), batch_target.view(-1))

                    s_smooth_loss = 0.15 * torch.mean(torch.clamp(
                        self.mse(F.log_softmax(p[:, :, 1:], dim=1), F.log_softmax(p.detach()[:, :, :-1], dim=1)), min=0,
                        max=16) * mask[:, :, 1:])
                    loss += s_ce_loss
                    loss += s_smooth_loss
                    ce_loss[i] += s_ce_loss.item()
                    smooth_loss[i] += s_smooth_loss.item()


                epoch_loss += loss.item()
                loss.backward()
                optimizer.step()

                _, predicted = torch.max(ps.data[-1], 1)
                correct += ((predicted == batch_target).float() * mask[:, 0, :].squeeze(1)).sum().item()
                total += torch.sum(mask[:, 0, :]).item()

            nums = len(batch_gen.list_of_examples)
            pr_str = "[epoch %d]: loss = %f, ce1 = %f, ce2 = %f, ce3 = %f, ce4 = %f, " \
                     "sm1 = %f, sm2 = %f, sm3 = %f, sm4 = %f, bd = %f, acc = %f" % \
                     (epoch + 1, epoch_loss / nums, np.round(ce_loss[0] / nums, 3), np.round(ce_loss[1] / nums, 3),
                      np.round(ce_loss[2] / nums, 3), np.round(ce_loss[3] / nums, 3),
                      np.round(smooth_loss[0] / nums, 3),
                      np.round(smooth_loss[1] / nums, 3), np.round(smooth_loss[2] / nums, 3),
                      np.round(smooth_loss[3] / nums, 3), np.round(boundary_loss / nums, 3),
                      float(correct) / total)
            self.log.logger.info(pr_str)

            # first update knn
            if epoch >= args.start:
                cm, cfeats = self.adjust_param(epoch)
                self.knn.update(cfeats)
                all_lambda.append(deepcopy(self.hyb_loss.lambdas))
                all_adjust.append(deepcopy(self.hyb_loss.adjustment.cpu().numpy()))
                all_epoch.append(epoch)
                all_cm.append(cm)

            # then calculate ncm results
            test_score, test_acc, test_score_ncm, test_acc_ncm = self.test(epoch, actions_dict, device)
            if test_score > best_score:
                best_score = test_score
                best_save = {'net': self.model.state_dict(), 'epoch': epoch, 'score': best_score}
                torch.save(best_save, save_dir + "/epoch-best" + ".model")
                self.log.logger.info("Save for the best model")

            if test_score_ncm > best_score_ncm:
                best_score_ncm = test_score_ncm
                best_save_ncm = {'net': self.model.state_dict(), 'epoch': epoch, 'score': best_score_ncm}
                torch.save(best_save_ncm, save_dir + "/epoch-best_ncm" + ".model")
                self.log.logger.info("Save for the best ncm model")

            last_save = {'net': self.model.state_dict(), 'epoch': epoch, 'score': best_score,
                         'score_ncm': best_score_ncm}
            torch.save(last_save, save_dir + "/epoch-last" + ".model")

        import pickle
        out_dict = {'lambda': all_lambda, 'cm': all_cm, 'epoch': all_epoch, 'adjust': all_adjust, 'prior': trans_prior}
        with open(save_dir + "/cm_lambda.pickle", 'wb') as f:
            pickle.dump(out_dict, f, pickle.HIGHEST_PROTOCOL)

    def adjust_param(self, epoch):
        self.model.eval()
        feats_all, labels_all = [], []
        total_frames = 0
        confusion_matrix = np.zeros((num_classes, num_classes, num_classes+1))
        with torch.no_grad():
            for _, items in enumerate(train_ad_loader):
                batch_input, batch_target, mask, vids = items
                batch_input, batch_target, mask = batch_input.to(device), batch_target.to(device), mask.to(device)
                total_frames += len(batch_target.view(-1))

                ps, features = self.model(batch_input, mask)

                feats_all.append(features.squeeze(0).cpu().numpy().T)
                labels_all.append(batch_target.view(-1, 1).cpu().numpy())

                _, predicted = torch.max(ps[-1], 1)
                previous = find_previous_seg(batch_target.view(-1))
                previous = previous.view(-1, 1).cpu().long().numpy()
                predicted, lbls = predicted.view(-1, 1).detach().cpu().numpy(), batch_target.view(-1, 1).cpu().numpy()
                np.add.at(confusion_matrix, (lbls, predicted, previous), 1)

        confusion_matrix /= total_frames
        self.hyb_loss.update(confusion_matrix)

        feats = np.concatenate(feats_all)
        labels = np.concatenate(labels_all).flatten()
        featmean = feats.mean(axis=0)

        def get_centroids(feats_, labels_):
            centroids, label_map = [], []
            for i in np.unique(labels_):
                centroids.append(np.mean(feats_[labels_ == i], axis=0))
                label_map.append(i)
            return np.stack(centroids), np.array(label_map)

        # Get unnormalized centorids
        un_centers, label_map = get_centroids(feats, labels)
        # Get l2n centorids
        l2n_feats = torch.Tensor(feats.copy())
        norm_l2n = torch.norm(l2n_feats, 2, 1, keepdim=True)
        l2n_feats = l2n_feats / norm_l2n
        l2n_centers, label_map = get_centroids(l2n_feats.numpy(), labels)
        # Get cl2n centorids
        cl2n_feats = torch.Tensor(feats.copy())
        cl2n_feats = cl2n_feats - torch.Tensor(featmean)
        norm_cl2n = torch.norm(cl2n_feats, 2, 1, keepdim=True)
        cl2n_feats = cl2n_feats / norm_cl2n
        cl2n_centers, label_map = get_centroids(cl2n_feats.numpy(), labels)

        self.model.train()
        return confusion_matrix, {'mean': featmean, 'uncs': un_centers, 'l2ncs': l2n_centers, 'cl2ncs': cl2n_centers,
                'label_map': label_map}


    def test(self, epoch, actions_dict, device):
        self.model.eval()
        preds, preds_ncm = [], []
        labels = []
        unique_label = []
        with torch.no_grad():
            for i, items in enumerate(test_loader):
                batch_input, batch_target,  mask,  vids = items
                batch_input, batch_target, mask = batch_input.to(device), batch_target.to(device), mask.to(device)
                unique_label = list(set(unique_label + list(batch_target.view(-1).cpu().numpy())))

                ps, features = self.model(batch_input, mask)
                _, predicted = torch.max(ps[-1], 1)
                predicted = predicted.squeeze().cpu().numpy()

                logits = self.knn(features.transpose(2, 1).squeeze(0))
                Y_ = logits.unsqueeze(0).transpose(2, 1)
                _, predicted_ncm = torch.max(Y_, 1)
                predicted_ncm = predicted_ncm.squeeze().detach().cpu().numpy()
                predicted_ncm = postprocess(predicted, predicted_ncm, ps[-1].transpose(2, 1).squeeze())

                batch_target = batch_target.squeeze().cpu().numpy()
                predicted_word, predicted_ncm_word, label_word = [], [], []
                for i in range(len(predicted)):
                    predicted_word += [label_dict[predicted[i].item()]] * self.sample_rate
                    predicted_ncm_word += [label_dict[predicted_ncm[i].item()]] * self.sample_rate
                    label_word += [label_dict[batch_target[i].item()]] * self.sample_rate

                preds.append(np.array(predicted_word))
                preds_ncm.append(np.array(predicted_ncm_word))
                labels.append(np.array(label_word))

            def cal_result(preds, labels):
                results = {}
                # global
                results['f1_10_g'] = overlap_f1(preds, labels, overlap=0.1)
                results['f1_25_g'] = overlap_f1(preds, labels, overlap=0.25)
                results['f1_50_g'] = overlap_f1(preds, labels, overlap=0.50)
                results['f_acc_g'] = accuracy(preds, labels).item()
                results['edit'] = edit_score(preds, labels)

                # balanced
                results['f1_10_s'] = overlap_f1_macro(preds, labels, overlap=0.1)
                results['f1_25_s'] = overlap_f1_macro(preds, labels, overlap=0.25)
                results['f1_50_s'] = overlap_f1_macro(preds, labels, overlap=0.50)
                results['f_rec'], results['f_prec'] = b_accuracy(preds, labels)

                valid_num = len(unique_label)
                results['f_acc'] = np.sum(results['f_rec']) / valid_num
                results['f1_10'] = np.sum(results['f1_10_s']) / valid_num
                results['f1_25'] = np.sum(results['f1_25_s']) / valid_num
                results['f1_50'] = np.sum(results['f1_50_s']) / valid_num
                results['bal_acc'] = results['f_acc']

                results['total_score'] = (results['f1_10_g'] + results['f1_25_g'] + results['f1_50_g']) / 3.0 +  results['f_acc_g'] \
                                         + results['edit'] + \
                                         (results['f1_10'] + results['f1_25'] + results['f1_50']) / 3.0 + results['bal_acc']

                self.log.logger.info(
                    "---[epoch %d]---: tst edit = %f, f1_10 = %f, f1_25 = %f, f1_50 = %f, acc = %f, total = %f "
                    % (epoch + 1, results['edit'], results['f1_10_g'], results['f1_25_g'], results['f1_50_g'],
                       results['f_acc_g'], results['total_score']))

                self.log.logger.info(" balanced acc = %f, f1_10 = %f, f1_25 = %f, f1_50 = %f" %
                                   (results['f_acc'], results['f1_10'], results['f1_25'], results['f1_50']))
                return results

            results = cal_result(preds, labels)
            results_ncm = cal_result(preds_ncm, labels)

        self.model.train()

        return results['total_score'], results['bal_acc'], results_ncm['total_score'], results_ncm['bal_acc']

    def predict(self, model_dir, results_dir, features_path, batch_gen_tst, actions_dict, sample_rate, device):

        self.model.eval()
        with torch.no_grad():
            self.model.to(device)
            state = torch.load(model_dir + "/epoch-best" + ".model", map_location=device)
            self.model.load_state_dict(state['net'])

            import time

            time_start = time.time()
            for i, items in enumerate(test_loader):
                batch_input, batch_target, mask,  vids = items
                vid = vids[0]
                features = np.load(features_path + vid.split('.')[0] + '.npy')
                features = features[:, ::sample_rate]

                input_x = torch.tensor(features, dtype=torch.float)
                input_x.unsqueeze_(0)
                input_x = input_x.to(device)
                predictions, _ = self.model(input_x, torch.ones(input_x.size(), device=device))

                _, predicted = torch.max(predictions[-1].data, 1)
                predicted = predicted.squeeze()
                recognition = []
                for i in range(len(predicted)):
                    recognition = np.concatenate((recognition, [list(actions_dict.keys())[
                                                                    list(actions_dict.values()).index(
                                                                        predicted[i].item())]] * sample_rate))
                f_name = vid.split('/')[-1].split('.')[0]
                f_ptr = open(results_dir + "/" + f_name, "w")
                f_ptr.write("### Frame level recognition: ###\n")
                f_ptr.write(' '.join(recognition))
                f_ptr.close()
            time_end = time.time()

    def get_knncentroids(self):
        self.model.eval()
        feats_all, labels_all = [], []
        with torch.set_grad_enabled(False):
            for i, items in enumerate(train_ad_loader):
                batch_input, batch_target, mask, vids = items
                batch_input, batch_target, mask = batch_input.to(device), batch_target.to(device), mask.to(device)

                predictions, features = self.model(batch_input, mask)

                feats_all.append(features.squeeze(0).cpu().numpy().T)
                labels_all.append(batch_target.view(-1, 1).cpu().numpy())

        feats = np.concatenate(feats_all)
        labels = np.concatenate(labels_all).flatten()

        featmean = feats.mean(axis=0)

        def get_centroids(feats_, labels_):
            centroids, label_map = [], []
            for i in np.unique(labels_):
                centroids.append(np.mean(feats_[labels_ == i], axis=0))
                label_map.append(i)
            return np.stack(centroids), np.array(label_map)

        # Get unnormalized centorids
        un_centers, label_map = get_centroids(feats, labels)

        # Get l2n centorids
        l2n_feats = torch.Tensor(feats.copy())
        norm_l2n = torch.norm(l2n_feats, 2, 1, keepdim=True)
        l2n_feats = l2n_feats / norm_l2n
        l2n_centers, label_map = get_centroids(l2n_feats.numpy(), labels)

        # Get cl2n centorids
        cl2n_feats = torch.Tensor(feats.copy())
        cl2n_feats = cl2n_feats - torch.Tensor(featmean)
        norm_cl2n = torch.norm(cl2n_feats, 2, 1, keepdim=True)
        cl2n_feats = cl2n_feats / norm_cl2n
        cl2n_centers, label_map = get_centroids(cl2n_feats.numpy(), labels)
        self.model.train()

        return {'mean': featmean, 'uncs': un_centers, 'l2ncs': l2n_centers, 'cl2ncs': cl2n_centers,
                'label_map': label_map}

    def predict_ncm(self, model_dir, results_dir, features_path, batch_gen_tst, actions_dict, sample_rate, device):

        self.model.to(device)
        state = torch.load(model_dir + "/epoch-best_ncm" + ".model", map_location=device)
        self.model.load_state_dict(state['net'])
        cfeats = self.get_knncentroids()
        self.knn.update(cfeats)
        self.model.eval()
        with torch.no_grad():
            import time

            time_start = time.time()
            for i, items in enumerate(test_loader):
                batch_input, batch_target, mask, vids = items
                vid = vids[0]
                features = np.load(features_path + vid.split('.')[0] + '.npy')
                features = features[:, ::sample_rate]

                input_x = torch.tensor(features, dtype=torch.float)
                input_x.unsqueeze_(0)
                input_x = input_x.to(device)
                ps, features = self.model(input_x, torch.ones(input_x.size(), device=device))
                _, cls_pred = torch.max(ps.data[-1], 1)
                cls_pred = cls_pred.view(-1).detach().cpu().numpy()

                logits = self.knn(features.transpose(2, 1).squeeze(0))
                Y_ = logits.unsqueeze(0).transpose(2, 1)
                _, predicted = torch.max(Y_, 1)
                predicted = predicted.squeeze().detach().cpu().numpy()
                predicted = postprocess(cls_pred, predicted, ps[-1].transpose(2, 1).squeeze())
                recognition = []
                for i in range(len(predicted)):
                    recognition = np.concatenate((recognition, [list(actions_dict.keys())[
                                                                    list(actions_dict.values()).index(
                                                                        predicted[i])]] * sample_rate))
                f_name = vid.split('/')[-1].split('.')[0]
                f_ptr = open(results_dir + "/" + f_name, "w")
                f_ptr.write("### Frame level recognition: ###\n")
                f_ptr.write(' '.join(recognition))
                f_ptr.close()
            time_end = time.time()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()
parser.add_argument('--action', default='train')
parser.add_argument('--dataset', default="breakfast")
parser.add_argument('--seed', default='42')
parser.add_argument('--split', default='1')
parser.add_argument('--resume', default=0, type=int, help='do we resume form lastest saved model')
parser.add_argument('--lr', default=0.01, type=float)
parser.add_argument('--tau', default=0.5, type=float)
parser.add_argument('--start', default=0, type=int)
parser.add_argument('--clip', default=0.9, type=float)
parser.add_argument('--suf', default='')
args = parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


seed = int(args.seed)

TYPE = '/mstcn_csl-t{}-l{}-c{}_{}'.format(args.tau, args.lr, args.clip, args.seed)

logpath = "results/" + args.dataset + "/split_{}/".format(args.split) + TYPE
logfile = logpath + '/' + time.strftime('%Y%m%d%H%M', time.localtime(time.time())) + '.log'
log = Logger(logpath, logfile, fmt="[%(asctime)s - %(levelname)s]: %(message)s")
log.logger.info('########################## MS-TCN #####################################')

log.logger.info("Training for MS-TCN with transition constraints by reweighting + SNCM")
log.logger.info(args)

num_stages = 4
num_layers = 10
num_f_maps = 64
features_dim = 2048
bz = 1
lr = 0.0005
num_epochs = 50

sample_rate = 1
boundary_smooth = 1
# sample input features @ 15fps instead of 30 fps for 50salads, and up-sample the output to 30 fps
if args.dataset == "50salads":
    sample_rate = 2
    boundary_smooth = 20

if args.dataset == "breakfast":
    boundary_smooth = 3

vid_list_file = "../data/" + args.dataset + "/splits/train.split" + args.split + ".bundle"
vid_list_file_tst = "../data/" + args.dataset + "/splits/test.split" + args.split + ".bundle"
features_path = "../data/" + args.dataset + "/features/"
gt_path = "../data/" + args.dataset + "/groundTruth/"
mapping_file = "../data/" + args.dataset + "/mapping.txt"

model_dir = "./models/" + args.dataset + "/split_" + args.split + TYPE
results_dir = "./results/" + args.dataset + "/split_" + args.split + TYPE

if not os.path.exists(model_dir):
    os.makedirs(model_dir)
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

file_ptr = open(mapping_file, 'r')
actions = file_ptr.read().split('\n')[:-1]
file_ptr.close()
actions_dict, label_dict = dict(), dict()
for a in actions:
    actions_dict[a.split()[1]] = int(a.split()[0])
    label_dict[int(a.split()[0])] = a.split()[1]

num_classes = len(actions_dict)

frame_per_cls = get_frame_num_per_cls(vid_list_file, gt_path, actions_dict)
class_prior = frame_per_cls / np.sum(frame_per_cls)
trans_prior = get_transition_prior(vid_list_file, gt_path, actions_dict)

trainer = Trainer(MultiStageModel, log, sample_rate, num_stages=num_stages, num_layers=num_layers,
                  num_f_maps=num_f_maps,
                  dim=features_dim, num_classes=num_classes)

batch_gen_ad = BatchGenerator(num_classes, actions_dict, gt_path, features_path, sample_rate, vid_list_file)
train_ad_loader = torch.utils.data.DataLoader(dataset=batch_gen_ad, batch_size=1, shuffle=False, pin_memory=False,
                                              num_workers=2)

if args.action == "train":
    batch_gen = BatchGenerator(num_classes, actions_dict, gt_path, features_path, sample_rate, vid_list_file)
    batch_gen_tst = BatchGenerator(num_classes, actions_dict, gt_path, features_path, sample_rate, vid_list_file_tst)

    train_loader = torch.utils.data.DataLoader(dataset=batch_gen, batch_size=1, shuffle=True, pin_memory=True,
                                               num_workers=2)
    test_loader = torch.utils.data.DataLoader(dataset=batch_gen_tst, batch_size=1, shuffle=False, pin_memory=False,
                                              num_workers=2)
    trainer.train(model_dir, num_epochs=num_epochs, batch_size=bz, learning_rate=lr, device=device,
                  actions_dict=actions_dict, batch_gen_tst=batch_gen_tst)

if args.action == "predict":
    batch_gen_tst = BatchGenerator(num_classes, actions_dict, gt_path, features_path, sample_rate, vid_list_file_tst)
    test_loader = torch.utils.data.DataLoader(dataset=batch_gen_tst, batch_size=1, shuffle=True, pin_memory=False,
                                              num_workers=2)
    if args.suf == '':
        if not os.path.exists(os.path.join(results_dir, 'prediction')):
            os.makedirs(os.path.join(results_dir, 'prediction'))
        trainer.predict(model_dir, os.path.join(results_dir, 'prediction'), features_path, batch_gen_tst, actions_dict,
                        sample_rate, device)
    else:
        if not os.path.exists(os.path.join(results_dir + '_ncm', 'prediction')):
            os.makedirs(os.path.join(results_dir + '_ncm', 'prediction'))
        trainer.predict_ncm(model_dir, os.path.join(results_dir + '_ncm', 'prediction'), features_path, batch_gen_tst, actions_dict,
                        sample_rate, device)




