import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

st.title('人工智慧分類器')
# 側邊欄
data_name = st.sidebar.selectbox(
    '請選擇資料集', ['iris','wine','cancer']
)
classifier = st.sidebar.selectbox(
    '請選擇分類模型', ['KNN','SVM','RandomForest']
)

# 下載資料集
def loadData(name):
    data = None
    if name =='iris':
        data = datasets.load_iris()
    elif name =='wine':
        data = datasets.load_wine()
    else:
        data = datasets.load_breast_cancer()

    X = data.data
    y = data.target
    return X,y

# 顯示資料集資訊
X, y = loadData(data_name)
st.write('## 資料集結構', X.shape)
st.write('## 資料集分類', len(np.unique(y)))

# 定義模型參數
def parameter(clf):
    p = {}
    if clf == 'SVM':
        C = st.sidebar.slider('C', 0.1, 10.0)
        p['C'] = C
    elif clf == 'KNN':
        K = st.sidebar.slider('K', 1, 20)
        p['K'] = K
    else:
        max_depth = st.sidebar.slider('max_depth', 1, 20)
        p['dep'] = max_depth
        trees = st.sidebar.slider('n_estimator', 10, 500)
        p['trees'] = trees
    return p

# 取得參數
params = parameter(classifier)

# 建立分類器模型
def getClassified(clf, p):
    now_clf = None
    if clf == 'SVM':
        now_clf = SVC(C = params['C'])
    elif clf == 'KNN':
        now_clf = KNeighborsClassifier(n_neighbors = params['K'])
    else:
        now_clf = RandomForestClassifier(n_estimators = params['trees'],
                                         max_depth = params['dep'],
                                         random_state = 123)
    return now_clf

# 取得物件模型
clf = getClassified(classifier, params)

# 分割資料集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123, stratify=y)

# 訓練資料集
clf.fit(X_train, y_train)

# 預測
y_pred = clf.predict(X_test)

# 評估準確率
acc = accuracy_score(y_test, y_pred)

# 顯示結果
st.write('## 準確率:', acc)

# PCA降維
pca = PCA(2)
new_X = pca.fit_transform(X)

#作圖
X1 = new_X[:,0]
X2 = new_X[:,1]

fig = plt.figure()
plt.scatter(X1, X2, c=y, alpha=0.7)
plt.xlabel('PCA-1')
plt.ylabel('PCA-2')
plt.title('PCA plot')

st.pyplot(fig)