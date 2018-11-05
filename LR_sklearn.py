# coding=utf-8


import numpy as np
import time
import pandas as pd
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression

# 导入数据
dataset = pd.read_csv('datasets/mnist_test.csv', delimiter=',', header=None).values
X = dataset[:, 1:]
y = dataset[:, 0]
X = X.astype('float')/256
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2, random_state=0)


start = time.clock()
# 初始化模型
lr_clf = LogisticRegression()
# 训练
lr_clf.fit(X_train, y_train)
# 测试——训练集评分
score_test = lr_clf.score(X_test, y_test)
# 测试——测试集评分
score_train = lr_clf.score(X_train, y_train)

print("训练集得分: " + str(score_train))
print("测试集得分: " + str(score_test))


# end time
elapsed = (time.clock() - start)
print("Time used:", elapsed)
