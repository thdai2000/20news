import numpy as np
from pre_processing import normalization as nm
import os
from prettytable import PrettyTable
import math


def get_list_of_vectors():
    path = 'data_2/v_train'
    D = []
    for dir in os.listdir(path):
        cur_path = os.path.join(path, dir)
        for file in os.listdir(cur_path):
            vec = nm.txt_to_dic(os.path.join(cur_path, file))
            D.append(vec)
    return D


def get_idf():
    D = get_list_of_vectors()
    V = nm.txt_to_dic('data_2/dic_all.txt')
    V_idf = {}
    len_D = len(D)
    for t in V:
        df = 0
        for d in D:
            if t in d.keys():
                df += 1
        # print(df)
        V_idf[t] = math.log(float(len_D)/(df + 1), 10)
        # print('word ' + t + ' done.')
    return V_idf


def print_confusion_matrix(categories, tp, tn, fp, fn):
    t = PrettyTable(['Actual \\ Predict', categories[0], categories[1]])
    t.add_row([categories[0], tn, fp])
    t.add_row([categories[1], fn, tp])
    return t


def get_d_array(path, V):
    d = nm.txt_to_dic(path)
    l = np.zeros(len(V))
    for t in V:
        # print(t)
        if t in d.keys():
            # l[int(t)] = (float(d[t])/len(d))*V_idf[t]
            l[int(t)] = int(d[t])
    return l


def get_data_and_labels(which_data, V, categories):
    path = which_data + '/v_test'
    data = []
    labels = []
    C = {1: categories[0], -1: categories[1]}
    for dir in os.listdir(path):
        cur_path = os.path.join(path, dir)
        if dir == C[1]:
            for file in os.listdir(cur_path):
                data.append(get_d_array(os.path.join(cur_path, file), V))
                labels.append(1)
        elif dir == C[-1]:
            for file in os.listdir(cur_path):
                data.append(get_d_array(os.path.join(cur_path, file), V))
                labels.append(-1)
    return data, labels


def perceptron_train(which_data, categories, max_iter, a):
    V = nm.txt_to_dic(which_data + '/dic_all.txt')
    # V = get_idf()
    data, labels = get_data_and_labels(which_data, V, categories)
    # print('len(labels): ', len(labels))
    # print('len(data): ', len(data))
    w = np.zeros(len(V)) # w是|V|维向量，初始化为全0
    b = 0
    # 循环max_iter次
    for iter in range(max_iter):
        # 遍历所有训练数据
        for i in range(len(data)):
            x = data[i] # 一次读取1个文档作为输入
            y = labels[i]
            if y * (np.dot(x, w) + b) <= 0:
                delta = np.multiply(a*y, x)
                w = np.add(w, delta)
                b += a*y
    return w, b, V


def perceptron_predict(w, b, d):
    return np.sign(np.dot(d, w) + b)


def perceptron_test(which_data, categories, max_iter, a):
    print('\r'+'******************** Perceptron ********************')
    w, b, V = perceptron_train(which_data, categories, max_iter, a)
    tp = tn = fp = fn = 0
    C = {1: categories[0], -1: categories[1]}
    for cname in os.listdir(which_data + '/v_test'):
        if cname == C[1]:
            dir_path = os.path.join(which_data + '/v_test', cname)
            for file in os.listdir(dir_path):
                d = os.path.join(dir_path, file)
                result = perceptron_predict(w, b, get_d_array(d, V))
                if result == 1:
                    tp += 1
                else:
                    fn += 1
        else:
            dir_path = os.path.join(which_data + '/v_test', cname)
            for file in os.listdir(dir_path):
                d = os.path.join(dir_path, file)
                result = perceptron_predict(w, b, get_d_array(d, V))
                if result == -1:
                    tn += 1
                else:
                    fp += 1
    print('\r')
    print('Iteration Times: ', max_iter)
    print('Learning Rate: ', a)
    print("Confusion matrix: ")
    print(print_confusion_matrix([C[-1], C[1]], tp, tn, fp, fn))
    print("Precision: ", float(tp) / (tp + fp) * 100, "%")
    print("Recall: ", float(tp) / (tp + fn) * 100, "%")
    print('F1-score: ', float(2*tp) / (2*tp + fp + fn) * 100, "%")