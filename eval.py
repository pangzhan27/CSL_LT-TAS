import numpy as np
import argparse
import json
import time
import os
# from sklearn.metrics import balanced_accuracy_score
# import sklearn.metrics as sm


def read_file(path):
    #os.system("pwd")
    with open(path, 'r') as f:
        content = f.read()
        f.close()
    return content


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
        ends.append(i)
    return labels, starts, ends


def levenstein(p, y, norm=False):
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


def edit_score(recognized, ground_truth, norm=True, bg_class=["background"]):
    P, _, _ = get_labels_start_end_time(recognized, bg_class)
    Y, _, _ = get_labels_start_end_time(ground_truth, bg_class)
    return levenstein(P, Y, norm)


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


def func_eval(dataset, recog_path, file_list):
    ground_truth_path = "../data/" + dataset + "/groundTruth/"
    mapping_file = "../data/" + dataset + "/mapping.txt"
    list_of_videos = read_file(file_list).split('\n')[:-1]

    file_ptr = open(mapping_file, 'r')
    actions = file_ptr.read().split('\n')[:-1]
    file_ptr.close()
    actions_dict = dict()
    for a in actions:
        actions_dict[a.split()[1]] = int(a.split()[0])

    overlap = [.1, .25, .5]
    tp, fp, fn = np.zeros(3), np.zeros(3), np.zeros(3)

    correct = 0
    total = 0
    edit = 0

    for vid in list_of_videos:

        gt_file = ground_truth_path + vid
        gt_content = read_file(gt_file).split('\n')[0:-1]

        if 'c2f' in recog_path:
            recog_file = recog_path + vid
            recog_content = read_file(recog_file).split()
        else:
            recog_file = recog_path + vid.split('.')[0]
            recog_content = read_file(recog_file).split('\n')[1].split()

        for i in range(len(gt_content)):
            total += 1
            if gt_content[i] == recog_content[i]:
                correct += 1

        edit += edit_score(recog_content, gt_content)

        for s in range(len(overlap)):
            tp1, fp1, fn1 = f_score(recog_content, gt_content, overlap[s])
            tp[s] += tp1
            fp[s] += fp1
            fn[s] += fn1

    acc = 100 * float(correct) / total
    edit = (1.0 * edit) / len(list_of_videos)
    #     print("Acc: %.4f" % (acc))
    #     print('Edit: %.4f' % (edit))
    f1s = np.array([0, 0, 0], dtype=float)
    for s in range(len(overlap)):
        precision = tp[s] / float(tp[s] + fp[s])
        recall = tp[s] / float(tp[s] + fn[s])

        f1 = 2.0 * (precision * recall) / (precision + recall)

        f1 = np.nan_to_num(f1) * 100
        #         print('F1@%0.2f: %.4f' % (overlap[s], f1))
        f1s[s] = f1

    return acc, edit, f1s


###################################### macro balanced eval
def f_score_ana(recognized, ground_truth, overlap, actions_dict, bg_class=["background"]):
    tp = np.zeros(len(actions_dict))
    fp = np.zeros(len(actions_dict))
    fn = np.zeros(len(actions_dict))
    if len(recognized) == 0 and len(ground_truth)==0:
        return tp, fp, fn
    elif len(recognized) == 0:
        y_label, y_start, y_end = get_labels_start_end_time(ground_truth, bg_class)
        for j in range(len(y_label)):
            fn[actions_dict[y_label[j]]] += 1
        return tp, fp, fn
    elif len(ground_truth) == 0:
        p_label, p_start, p_end = get_labels_start_end_time(recognized, bg_class)
        for j in range(len(p_label)):
            fp[actions_dict[p_label[j]]] += 1
        return tp, fp, fn

    p_label, p_start, p_end = get_labels_start_end_time(recognized, bg_class)
    y_label, y_start, y_end = get_labels_start_end_time(ground_truth, bg_class)

    
    hits = np.zeros(len(y_label))

    for j in range(len(p_label)):
        intersection = np.minimum(p_end[j], y_end) - np.maximum(p_start[j], y_start)
        union = np.maximum(p_end[j], y_end) - np.minimum(p_start[j], y_start)
        IoU = (1.0*intersection / union)*([p_label[j] == y_label[x] for x in range(len(y_label))])
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


def overlap_f1_macro(P, Y, actions_dict, zero_ind, overlap=.1,  bg_class=["background"]):
    TP, FP, FN = 0, 0, 0
    for i in range(len(P)):
        tp, fp, fn = f_score_ana(P[i], Y[i], overlap, actions_dict, bg_class)
        TP += tp
        FP += fp
        FN += fn
    precision = TP / (TP + FP + 1e-8)
    recall = TP / (TP + FN + 1e-8)
    F1 = 2 * (precision * recall) / (precision + recall+1e-16)
    F1 = np.nan_to_num(F1)

    F1_nonzero = np.delete(F1, zero_ind, axis=0)

    g_pre = np.sum(TP) / (np.sum(TP) + np.sum(FP) + 1e-8)
    g_rec = np.sum(TP) / (np.sum(TP) + np.sum(FN) + 1e-8)
    g_F1 = 2 * (g_pre * g_rec) / (g_pre + g_rec + 1e-16)

    return g_F1 * 100, np.mean(F1_nonzero * 100), F1* 100

def b_accuracy_macro(P, Y, actions_dict, zero_ind):
    total = np.zeros(len(actions_dict))
    correct = np.zeros(len(actions_dict))
    fps = np.zeros(len(actions_dict))
    for i in range(len(P)):
        num = min(len(P[i]), len(Y[i]))
        for j in range(num):
            if P[i][j] == Y[i][j]:
                correct[actions_dict[Y[i][j]]] +=1
            total[actions_dict[Y[i][j]]] += 1
            fps[actions_dict[P[i][j]]] += 1

    acc = 100 * correct / (total+1e-3)
    acc_nonzero = np.delete(acc, zero_ind, axis=0)
    glob_acc = 100 * np.sum(correct) / np.sum(total)

    prec = 100 * correct / (fps+1e-3)
    prec_nonzero = np.delete(prec, zero_ind, axis=0)

    f1 = 2 * (prec * acc) / (prec + acc +1e-16)
    f1_nonzero = np.delete(f1, zero_ind, axis=0)
    return glob_acc, np.mean(acc_nonzero), acc, np.mean(prec_nonzero), prec,  np.mean(f1_nonzero), f1

def func_eval_balanced(dataset, recog_path, file_list):
    ground_truth_path = "../data/" + dataset + "/groundTruth/"
    mapping_file = "../data/" + dataset + "/mapping.txt"
    list_of_videos = read_file(file_list).split('\n')[:-1]

    file_ptr = open(mapping_file, 'r')
    actions = file_ptr.read().split('\n')[:-1]
    file_ptr.close()
    actions_dict = dict()
    for a in actions:
        actions_dict[a.split()[1]] = int(a.split()[0])

    overlap = [.1, .25, .5]
    P, Y = [], []
    for vid in list_of_videos:
        gt_file = ground_truth_path + vid
        gt_content = read_file(gt_file).split('\n')[0:-1]

        if 'c2f' in recog_path:
            recog_file = recog_path + vid
            recog_content = read_file(recog_file).split()
        else:
            recog_file = recog_path + vid.split('.')[0]
            recog_content = read_file(recog_file).split('\n')[1].split()

        Y.append(gt_content)
        P.append(recog_content)

    ### find the non-appear classes
    total = np.zeros(len(actions_dict))
    for i in range(len(Y)):
        for j in range(len(Y[i])):
            total[actions_dict[Y[i][j]]] = 1
    zero_ind = np.where(total == 0)[0]
    print('non-appeared classes: ', zero_ind)
    ##################
    g_acc, b_acc, acc_split, b_prec, prec_split, b_f1, f1_split = b_accuracy_macro(P, Y, actions_dict, zero_ind)
    #macro, split = acc_per_class(P, Y, actions_dict, zero_ind)

    g_f1s = np.array([0, 0, 0], dtype=float)
    b_f1s = np.array([0, 0, 0], dtype=float)
    f1s_split = [np.zeros(len(actions_dict)), np.zeros(len(actions_dict)), np.zeros(len(actions_dict)) ]
    for s in range(len(overlap)):
        g_f1, b_f1, f1_split = overlap_f1_macro(P, Y, actions_dict, zero_ind, overlap=overlap[s]) #, bg_class=['SIL']
        b_f1s[s] += b_f1
        g_f1s[s] += g_f1
        f1s_split[s] += f1_split

    return g_acc, g_f1s, b_acc, b_f1s, acc_split, f1s_split, total, b_prec, prec_split, b_f1, f1_split

def group_eval(acc_split, prec_split, f_split, f1_split, dataset, zero_inds):
    zero_index = np.where(zero_inds==0)[0]
    import json
    mapping_file = "../data/" + dataset + "/mapping.txt"
    file_ptr = open(mapping_file, 'r')
    actions = file_ptr.read().split('\n')[:-1]
    file_ptr.close()
    actions_dict = dict()
    for a in actions:
        actions_dict[a.split()[1]] = int(a.split()[0])

    with open('../data/{}_frame_bin2.json'.format(dataset), 'r') as f:
        group_dict = json.load(f)

    def group_results(cls_spllit):
        out = {}
        for ky in group_dict:
            out[ky] = []
            for item in group_dict[ky]:
                if actions_dict[item] not in zero_index:
                     out[ky].append(cls_spllit[actions_dict[item]])
        final_out = {ky: np.mean(out[ky]) for ky, vl in out.items()}
        return final_out

    acc_group = group_results(acc_split)
    f1_10 = group_results(f1_split[0])
    f1_25 = group_results(f1_split[1])
    f1_50 = group_results(f1_split[2])
    prec_group = group_results(prec_split)
    f1_group = group_results(f_split)
    return acc_group, f1_10, f1_25, f1_50, prec_group, f1_group

def main():
    cnt_split_dict = {
        '50salads': 5,
        'gtea': 4,
        'breakfast': 4
    }

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default="breakfast")
    parser.add_argument('--split', default=0, type=int)
    parser.add_argument('--result_dir', default='results')
    parser.add_argument('--backbone', default='mstcn')
    parser.add_argument('--method', default='mstcn_baseline_42')
    args = parser.parse_args()

    acc_all, edit_all, f1s_all = 0., 0., [0., 0., 0.]
    acc_all_b, prec_all_b, f1_all_b, f1s_all_b = 0., 0., 0., [0., 0., 0.]
    acc_split_bs, prec_split_bs, f1_split_bs, f1s_split_bs = 0., 0., 0.,  [0., 0., 0.]
    zero_inds = []

    ##################
    if args.split == 0:
        for split in range(1, cnt_split_dict[args.dataset] + 1):
            recog_path = "{}/{}/".format(args.backbone, args.result_dir) + args.dataset + "/split_{}/".format(split) + args.method + "/prediction" + "/"  
            file_list = "../data/" + args.dataset + "/splits/test.split{}".format(split) + ".bundle"

            #### traditional metric #####
            acc, edit, f1s = func_eval(args.dataset, recog_path, file_list)
            acc_all += acc
            edit_all += edit
            f1s_all[0] += f1s[0]
            f1s_all[1] += f1s[1]
            f1s_all[2] += f1s[2]

            ##### macro balanced metric #####
            acc_avg, f1s_avg, acc_b, f1s_b, acc_split_b, f1s_split_b, zero_ind, prec_b, prec_split_b, f1_b, f1_split_b  = func_eval_balanced(args.dataset, recog_path,
                                                                                                    file_list)
            zero_inds.append(zero_ind)
            acc_all_b += acc_b
            f1s_all_b[0] += f1s_b[0]
            f1s_all_b[1] += f1s_b[1]
            f1s_all_b[2] += f1s_b[2]
            acc_split_bs += acc_split_b
            f1s_split_bs[0] += f1s_split_b[0]
            f1s_split_bs[1] += f1s_split_b[1]
            f1s_split_bs[2] += f1s_split_b[2]

            prec_all_b += prec_b
            f1_all_b += f1_b
            prec_split_bs += prec_split_b
            f1_split_bs += f1_split_b
            assert acc_avg == acc
            
            ind = np.where(zero_ind == 0)[0]


        acc_all /= cnt_split_dict[args.dataset]
        edit_all /= cnt_split_dict[args.dataset]
        f1s_all = [i / cnt_split_dict[args.dataset] for i in f1s_all]

        acc_all_b /= cnt_split_dict[args.dataset]
        prec_all_b /= cnt_split_dict[args.dataset]
        f1_all_b /= cnt_split_dict[args.dataset]
        f1s_all_b = [i / cnt_split_dict[args.dataset] for i in f1s_all_b]
        zero_inds = np.array(zero_inds)
        zero_inds = np.sum(zero_inds, axis=0)
        # print(zero_inds)
        acc_split_bs /= zero_inds
        f1s_split_bs = [i / zero_inds for i in f1s_split_bs]
        prec_split_bs /= zero_inds
        f1_split_bs /= zero_inds
  
    else:
        for split in range(args.split, args.split+ 1):
            recog_path = "{}/{}/".format(args.backbone, args.result_dir) + args.dataset + "/split_{}/".format(split) + args.method + "/prediction" + "/"  
            file_list = "../data/" + args.dataset + "/splits/test.split{}".format(split) + ".bundle"

            #### traditional metric #####
            acc, edit, f1s = func_eval(args.dataset, recog_path, file_list)
            acc_all += acc
            edit_all += edit
            f1s_all[0] += f1s[0]
            f1s_all[1] += f1s[1]
            f1s_all[2] += f1s[2]

            ##### macro balanced metric #####
            acc_avg, f1s_avg, acc_b, f1s_b, acc_split_b, f1s_split_b, zero_ind, prec_b, prec_split_b, f1_b, f1_split_b  = func_eval_balanced(args.dataset, recog_path,
                                                                                                    file_list)
            zero_inds.append(zero_ind)
            acc_all_b += acc_b
            f1s_all_b[0] += f1s_b[0]
            f1s_all_b[1] += f1s_b[1]
            f1s_all_b[2] += f1s_b[2]
            acc_split_bs += acc_split_b
            f1s_split_bs[0] += f1s_split_b[0]
            f1s_split_bs[1] += f1s_split_b[1]
            f1s_split_bs[2] += f1s_split_b[2]

            prec_all_b += prec_b
            f1_all_b += f1_b
            prec_split_bs += prec_split_b
            f1_split_bs += f1_split_b
            assert acc_avg == acc
            
            ind = np.where(zero_ind == 0)[0]


        acc_all /= 1
        edit_all /= 1
        f1s_all = [i / 1 for i in f1s_all]

        acc_all_b /= 1
        prec_all_b /= 1
        f1_all_b /= 1
        f1s_all_b = [i / 1 for i in f1s_all_b]
        zero_inds = np.array(zero_inds)
        zero_inds = np.sum(zero_inds, axis=0)
        # print(zero_inds)
        acc_split_bs /= (zero_inds +1e-8)
        f1s_split_bs = [i / (zero_inds +1e-8) for i in f1s_split_bs]
        prec_split_bs /= (zero_inds +1e-8)
        f1_split_bs /= (zero_inds +1e-8)


    acc_group, f1_10, f1_25, f1_50, prec_group, f1_group = group_eval(acc_split_bs, prec_split_bs, f1_split_bs, f1s_split_bs, args.dataset, zero_inds)

    ### write to csv
    import pandas as pd
    import os
    head = True
    if os.path.exists('{}_summary.csv'.format(args.dataset)):
        head = False
    ########## micro
    pd.set_option("display.precision", 1)
    from scipy.stats import hmean, gmean
    micro_results = {'method': args.method, 
                      'bal_rec': np.mean(acc_split_bs), 'bal_f1@25': np.mean(f1s_split_bs[1]), 'edit score': edit_all, 
                      'bal_f1@10': np.mean(f1s_split_bs[0]), 'bal_f1@50': np.mean(f1s_split_bs[2]),
                      'glob_acc': acc_all, 'glob_f1@10': f1s_all[0], 'glob_f1@25': f1s_all[1], 'glob_f1@50': f1s_all[2], 
                      'head_acc': acc_group['head'], 'tail_acc': acc_group['tail'],  
                      'head_f1@25': f1_25['head'], 'tail_f1@25': f1_25['tail'],  
                      'head_f1@10': f1_10['head'], 'tail_f1@10': f1_10['tail'],  
                      'head_f1@50': f1_50['head'], 'tail_f1@50': f1_50['tail'], }
    
    
    frame_micro = pd.DataFrame([micro_results])
    frame_micro.to_csv('{}_summary.csv'.format(args.dataset), mode='a', index=False, header=head, float_format='%.1f')




if __name__ == '__main__':
    main()