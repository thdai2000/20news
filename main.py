from pre_processing import normalization as nm
import my_bayes
import my_perceptron
import my_knn
import my_svm
import os
import argparse


# 获得大词典并转储为dic_all.txt
def get_dic_all_and_save(which_data):
    dic_all = nm.get_dictionary(which_data + '/train')
    nm.dic_to_txt(dic_all, which_data + '/dic_all.txt')


# 文档转储为向量（按索引排序）
def transfer_raw_data_to_vectors(which_data):
    dic_all = nm.txt_to_dic(which_data + '/dic_all.txt')
    for which_set in ["train", "test"]:
        from_path = which_data + '/' + which_set
        to_path = which_data + '/v_' + which_set
        for dirname in os.listdir(from_path):
            from_dir_path = from_path + '/' + dirname
            to_dir_path = to_path + '/' + dirname
            if not os.path.exists(to_dir_path):
                os.makedirs(to_dir_path)
            print('loading ' + from_dir_path + '...')
            for filename in os.listdir(from_dir_path):
                sorted_vec = nm.get_sorted_vector(from_dir_path + '/' + filename, dic_all)
                nm.dic_to_txt(sorted_vec, to_dir_path + '/' + filename)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Text Classification')
    parser.add_argument('--class_num', type=int, default=3, help='Number of classes, either 2, 3, 10 or 20, default 3')
    parser.add_argument('--model', type=str, default="bayes", help='Model to use, either bayes, knn, perceptron or svm')
    opt = parser.parse_args()

    # 指定数据集
    which_data = 'data_' + str(opt.class_num)

    # 预处理，生成词集和文档的向量表示
    if not os.path.exists(which_data + "/dic_all.txt"):
        get_dic_all_and_save(which_data)
    if not os.path.exists(which_data + "/v_test") and not os.path.exists(which_data + "/v_train"):
        transfer_raw_data_to_vectors(which_data)

    if opt.model == "bayes":
        my_bayes.naive_bayes_test(which_data=which_data, lamb=0.2)
    if opt.model == "knn":
        my_knn.knn_test(which_data=which_data, p=2, k=1)
    if opt.model == "perceptron":
        my_perceptron.perceptron_test(which_data='data_2', categories=['comp.graphics', 'alt.atheism'], max_iter=11, a=1)
    if opt.model == "svm":
        my_svm.svm_test(category=['comp.graphics', 'alt.atheism'], C=1, kernel='linear')
