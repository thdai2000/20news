import numpy as np
from collections import Counter
from pre_processing import normalization as nm
import os
from prettytable import PrettyTable


def get_class_id(path):
    C = {}
    i = 0
    for cname in os.listdir(path):
        C[i] = cname
        i += 1
    return C


def print_prf_matrix(C, precision, recall, f1):
    t = PrettyTable(['\\', 'Precision', 'Recall', 'F1-score'])
    for i in range(len(C)):
        t.add_row([C[i], precision[i], recall[i], f1[i]])
    return t


def get_vd(path):
    d = nm.txt_to_dic(path)
    l = []
    for key, value in d.items():
        l.append([int(key), int(value)])
    return l


def get_data_and_labels(path):
    data = []
    labels = []
    C = get_class_id(path)
    for dir in os.listdir(path):
        cur_path = os.path.join(path, dir)
        for i in range(len(C)):
            if dir == C[i]:
                for file in os.listdir(cur_path):
                    data.append(get_vd(os.path.join(cur_path, file)))
                    labels.append(i)
    return data, labels


def get_dis(d, t, p):
    dis = 0
    p1 = p2 = 0
    len_d = len(d)
    len_t = len(t)
    while p1 < len_d and p2 < len_t:
        if d[p1][0] == t[p2][0]:
            dis += abs(d[p1][1]-t[p2][1]) ** p
            p1 += 1
            p2 += 1
        elif d[p1][0] < t[p2][0]:
            dis += d[p1][1] ** p
            p1 += 1
        else:
            dis += t[p2][1] ** p
            p2 += 1
    while p1 < len_d:
        dis += d[p1][1] ** p
        p1 += 1
    while p2 < len_t:
        dis += t[p2][1] ** p
        p2 += 1
    dis = dis ** 1/p
    return dis


# 构建KNN分类器
def knn_predict(d_path, train_data, labels, C, p, k):
    d = get_vd(d_path)
    dis = []
    for i in range(len(train_data)):
        dis.append(get_dis(d, train_data[i], p))

    dis = np.array(dis)

    sorted_index = dis.argsort()
    predict_classes = []
    for i in range(k):
        c = labels[sorted_index[i]]
        predict_classes.append(c)
    counter = Counter(predict_classes)
    max_c = counter.most_common(1)[0][0]
    print(C[max_c])
    return max_c


def knn_test(which_data, p, k):
    print('\r' + '******************** KNN ********************')
    C = get_class_id(which_data + '/v_train')
    # V = nm.txt_to_dic(which_data + '/dic_all.txt')
    train_data, labels = get_data_and_labels(which_data + '/v_train')

    t = {}.fromkeys(range(len(C)), 0)
    f = {}.fromkeys(range(len(C)), 0)
    pre = {}.fromkeys(range(len(C)), 0)
    for i in C:
        print('Test', C[i], '...')
        cur_path = which_data + '/v_test/' + C[i]
        for file in os.listdir(cur_path):
            pc = knn_predict(cur_path + '/' + file, train_data, labels, C, p, k)
            if pc == i:
                t[i] += 1
            else:
                f[i] += 1  # recall(i) = t[i]/(t[i]+f[i])
                pre[pc] += 1  # precision(i) = t[i]/(t[i] + pre[i])
    precision = {}
    recall = {}
    f1 = {}
    for i in range(len(C)):
        precision[i] = float(t[i]) / (t[i] + pre[i])
        recall[i] = float(t[i]) / (t[i] + f[i])
        f1[i] = float(2 * precision[i] * recall[i]) / (precision[i] + recall[i])
    print('P of Minkowski Distance: ', p)
    print('K of Knn: ', k)
    print(print_prf_matrix(C, precision, recall, f1))
    print('Macro-F1: ', float(sum(f1.values())) / len(C) * 100, '%')
    return