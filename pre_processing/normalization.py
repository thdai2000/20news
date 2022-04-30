from nltk.tokenize import word_tokenize
import re
import os
from collections import Counter


# 加载停用词
with open("pre_processing/stop_words.utf8", encoding='utf-8') as f:
    stopword_list = f.read().splitlines()


def get_list_of_text(filename):
    with open(filename, encoding='gb18030', errors='ignore') as f:
        text = f.read().splitlines()
    return text


def tokenize_text(text):
    tokens = word_tokenize(text)
    tokens = [token.lower() for token in tokens if bool(re.search('[a-z]', token))]
    return tokens


def remove_stopwords(text):
    tokens = tokenize_text(text)
    filtered_tokens = [token for token in tokens if token not in stopword_list]
    return filtered_tokens


def tokenize_document(document):
    tokenized_document = []
    for line in document:
        line = remove_stopwords(line)
        tokenized_document += line
    # tokenized_document = sorted(set(tokenized_document), key=tokenized_document.index)
    # print(tokenized_document)
    return tokenized_document


def list_to_dic(list):
    dic = {}
    for i in range(len(list)):
        dic[i] = list[i]
    return dic


def get_all_words(dir, words_list):
    for filename in os.listdir(dir):
        cur_path = dir + '/' + filename
        if os.path.isdir(cur_path):
            print('loading ' + cur_path + '...')
            tmp = get_all_words(cur_path, words_list)
            words_list += tmp
        else:
            doc = tokenize_document(get_list_of_text(cur_path))
            words_list += doc
    return words_list


def get_dictionary(dir):
    words_list = []
    for dirname in os.listdir(dir):
        cur_path = dir + '/' + dirname
        print('loading ' + cur_path + '...')
        for filename in os.listdir(cur_path):
            doc = tokenize_document(get_list_of_text(cur_path + '/' + filename))
            words_list += doc
    # words_list = get_all_words(dir, words_list)
    dic_list = list(set(words_list))
    dic = list_to_dic(dic_list)
    print('Finished getting dictionary.')
    return dic


def get_id(target_word, dic):
    for id, word in dic.items():
        if word == target_word:
            word_id = id
            break
    return word_id


def sorted_vector(vec):
    int_keys = []
    for i in range(len(list(vec.keys()))):
        int_keys.append(int(list(vec.keys())[i]))
    sorted_keys = sorted(int_keys)
    sorted_vec = {}
    for i in sorted_keys:
        sorted_vec[i] = vec[str(i)]
    return sorted_vec


# doc is name of document, dic_all is the dictionary of all training data
# return a dictionary with frequency
def get_sorted_vector(doc, dic_all):
    doc = tokenize_document(get_list_of_text(doc))
    dic_doc = {}
    for word in set(doc):
        if word in dic_all.values():
            dic_doc[get_id(word, dic_all)] = doc.count(word)
    dic_doc = sorted_vector(dic_doc)
    return dic_doc


def dic_to_txt(dic, path):
    with open(path, 'w+', encoding='utf-8') as f:
        for key, value in dic.items():
            # if isinstance(key, int):
            key = str(key)
            # if isinstance(value, int):
            value = str(value)
            f.write('<' + key + ',' + value + '>')
            f.write('\r')
    return f


def txt_to_dic(txt):
    dic = {}
    with open(txt, 'r', encoding='utf-8') as f:
        tuples = f.read().splitlines()
    for tuple in tuples:
        key = (re.search('<(.*?),', tuple)).group(1)
        val = (re.search(',(.*?)>', tuple)).group(1)
        dic[key] = val
    return dic


def value_to_int(dic):
    for key, value in dic.items():
        dic[key] = int(dic[key])
    return dic


def value_to_float(dic):
    for key, value in dic.items():
        dic[key] = float(dic[key])
    return dic


# O(n)=n^2
# path should be a directory
def merge_all_vec_in_sort(path):
    dic = Counter({})
    for vec in os.listdir(path):
        vec = os.path.join(path, vec)
        dic += Counter(value_to_int(txt_to_dic(vec)))
    dic = sorted_vector(dic)
    return dic


# path should be a directory
def get_sum_words(path, n):
    if os.path.isfile(path):
        vec = txt_to_dic(path)
        for freq in vec.values():
            n += int(freq)
        return n
    else:
        for file in os.listdir(path):
            cur_path = os.path.join(path, file)
            if os.path.isdir(cur_path):
                n = get_sum_words(cur_path, n)
            else:
                vec = txt_to_dic(cur_path)
                for freq in vec.values():
                    n += int(freq)
        return n