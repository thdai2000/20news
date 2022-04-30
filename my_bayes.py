from pre_processing import normalization as nm
import os
import math
import numpy as np
from prettytable import PrettyTable


def get_class_id(path):
    C = {}
    i = 0
    for cname in os.listdir(path):
        C[i] = cname[:-4]
        i += 1
    return C


def print_prf_matrix(C, precision, recall, f1):
    t = PrettyTable(['\\', 'Precision', 'Recall', 'F1-score'])
    for i in range(len(C)):
        t.add_row([C[i], precision[i], recall[i], f1[i]])
    return t


# 每个类目录转储为一个向量
def class_to_txt(which_data):
    from_path = which_data + '/v_train'
    to_path = which_data + '/bayesian_statistic/v_class'
    if not os.path.exists(to_path):
        os.makedirs(to_path)
    for dir in os.listdir(from_path):
        # print('loading ' + dir + '...')
        c_dir = from_path + '/' + dir
        v = nm.merge_all_vec_in_sort(c_dir)
        nm.dic_to_txt(v, to_path + '/' + dir + '.txt')
    return


def prior_to_txt(which_data, C, N):
    from_path = which_data + '/bayesian_statistic/v_class'
    to_path = which_data + '/bayesian_statistic/prior.txt'
    prior = {}
    for c in C:
        Nc = nm.get_sum_words(from_path + '/' + C[c] + '.txt', 0)
        prior[c] = Nc/N
    nm.dic_to_txt(prior, to_path)
    return


def condprob_to_npy(which_data, C, V, lamb):
    NV = len(V)
    condprob = [[0 for i in range(len(C))] for j in range(NV)]
    for c in C:
        vc_path = which_data + '/bayesian_statistic/v_class' + '/' + C[c] + '.txt'
        ck = nm.get_sum_words(vc_path, 0)
        vc = nm.txt_to_dic(vc_path)
        for t in V:
            # 计算类c下单词t的出现次数
            try:
                Tct = vc[t]
            except:
                Tct = 0
            condprob[int(t)][c] = (int(Tct) + lamb) / (ck + NV*lamb)  # =P(t|c)
    np.save(which_data + '/bayesian_statistic/condprob.npy', np.array(condprob))
    return


def naive_bayes_train(which_data, lamb):
    class_to_txt(which_data)
    print('Finished getting vectors of classes.')
    C = get_class_id(which_data + '/bayesian_statistic/v_class')
    N = nm.get_sum_words(which_data + '/bayesian_statistic/v_class', 0)
    V = nm.txt_to_dic(which_data + '/dic_all.txt')
    prior_to_txt(which_data, C, N)
    print('Finished getting prior probability.')
    condprob_to_npy(which_data, C, V, lamb)
    print('Finished getting conditional probability')
    return


# d为输入的文档，输出该文档的最大概率类别
def naive_bayes_predict(d, prior, condprob, C, V):
    vd = nm.txt_to_dic(d)
    # print(condprob)
    score = []
    for c in C.keys():
        score.append(math.log(prior[str(c)]))
        for t in V:
            if t in vd.keys():
                score[c] += math.log(condprob[int(t)][c]) # 等价于条件概率连乘*先验概率，得后验概率
    max_index = score.index(max(score))
    # print(C[max_index])
    return max_index # 后验概率最大化


def naive_bayes_test(which_data, lamb):
    print('\r' + '******************** Naive Bayes ********************')
    naive_bayes_train(which_data, lamb)
    prior = nm.value_to_float(nm.txt_to_dic(which_data + '/bayesian_statistic/prior.txt'))
    condprob = (np.load(which_data + '/bayesian_statistic/condprob.npy', allow_pickle=True)).tolist()
    C = get_class_id(which_data + '/bayesian_statistic/v_class')
    V = nm.txt_to_dic(which_data + '/dic_all.txt')
    dir = which_data + '/v_test'

    t = {}.fromkeys(range(len(C)), 0)
    f = {}.fromkeys(range(len(C)), 0)
    p = {}.fromkeys(range(len(C)), 0)
    for i in C:
        print('Test', C[i], '...')
        cur_path = dir + '/' + C[i]
        for file in os.listdir(cur_path):
            pc = naive_bayes_predict(cur_path + '/' + file, prior, condprob, C, V)
            if pc == i:
                t[i] += 1
            else:
                f[i] += 1 # recall(i) = t[i]/(t[i]+f[i])
                p[pc] += 1 # precision(i) = t[i]/(t[i] + p[i])
    precision = {}
    recall = {}
    f1 = {}
    for i in range(len(C)):
        precision[i] = float(t[i]) / (t[i] + p[i])
        recall[i] = float(t[i]) / (t[i]+f[i])
        f1[i] = float(2*precision[i]*recall[i]) / (precision[i] + recall[i])
    print('Lambda: ', lamb)
    print(print_prf_matrix(C, precision, recall, f1))
    print('Macro-F1: ', float(sum(f1.values())) / len(C) * 100, '%')

    return


# # 伪代码
# # C为类别集合，V为训练集的词典, N是训练集总字数
# nb_train(C,V, N){
#     for each c∈C
#         # Nc为类别c下的单词总数
#         Nc←CountTokensInClass(D,c)
#         prior[c]←Nc/N
#         # 将类别c下的文档连接成一个大字符串
#         textc←ConcatenateTextOfAllDocsInClass(D,c)
#         for each t∈V
#         # 计算类c下单词t的出现次数
#             Tct←CountTokensOfTerm(textc,t)
#         for each t∈V
#             condprob[t][c] += P(t|c)=（|tk∈C|+1）/（|ck|+|V|）
#     return V,prior,condprob
# }
#
# # d为输入的文档，输出该文档的最大概率类别
# nb_predict(C,V,prior,condprob,d) {
# # 将文档d中的单词抽取出来，允许重复，如果单词是全新的，在全局单词表V中都没出现过，则忽略掉
# W←ExtractTokensFromDoc(V,d)
#     for each c∈C
#         score[c]←prior[c]  # 先验概率
#         for each t∈W
#             if t∈Vd
#                 score[c] *= condprob[t][c] # 条件概率连乘*先验概率，得后验概率
#     return max(score[c]) # 后验概率最大化
# }
