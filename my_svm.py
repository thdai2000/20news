from sklearn.datasets import fetch_20newsgroups  # 获取数据集
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer  # TF-IDF文本特征提取
from sklearn.metrics import f1_score
from sklearn.svm import SVC


def svm_test(category, C, kernel):
    print('\r' + '******************** SVM ********************')
    train = fetch_20newsgroups(subset='train', categories=category)
    test = fetch_20newsgroups(subset='test', categories=category)

    vectorizer = TfidfVectorizer()
    v_train = vectorizer.fit_transform(train.data)
    v_test = vectorizer.transform(test.data)

    model = SVC(C=C, kernel=kernel)
    model.fit(v_train, train.target)

    y_true = test.target
    y_pred = model.predict(v_test)
    print('Penalty Constant: ', C)
    print('Kernel Function: ', kernel)
    print('Macro-F1: ', f1_score(y_true, y_pred, average='macro') * 100, '%')

    return
