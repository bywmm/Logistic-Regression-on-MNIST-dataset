# -*- coding: utf-8 -*-


import pandas as pd
import time
import numpy as np
import matplotlib.pyplot as plt
from LR_model import model
from sklearn import model_selection


# 读取数据
start = time.clock()
"""
train_set = pd.read_csv('datasets/mnist_train.csv',delimiter=',',header=None).values
X_train=train_set[:, 1:]
y_train=train_set[:, 0]
test_set = pd.read_csv('datasets/mnist_test.csv',delimiter=',',header=None).values
X_test=test_set[:, 1:]
y_test=test_set[:, 0]
X_train = X_train.astype('float')/256
X_test = X_test.astype('float')/256
"""
dataset = pd.read_csv('datasets/mnist_test.csv', delimiter=',', header=None).values
X = dataset[:, 1:]
y = dataset[:, 0]
X = X.astype('float')/256
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2, random_state=0)

print("数据读取成功！用时："+str(time.clock()-start))


"""
# 查看数据集的图片
plt.figure()
fig, ax = plt.subplots(
    nrows=2,
    ncols=5,
    sharex=True,
    sharey=True, )
 
ax = ax.flatten()
for i in range(10):
    img = X_train[i].reshape(28, 28)
    ax[i].imshow(img, cmap='Greys', interpolation='nearest')
 
ax[0].set_xticks([])
ax[0].set_yticks([])
plt.tight_layout()
"""

m_train = X_train.shape[0]
m_test = X_test.shape[0]
# X_train = X_train.T
# X_test = X_test.T
y_train = y_train.reshape(1, -1)
y_test = y_test.reshape(1, -1)
print("train_x shape: " + str(X_train.shape))
print("train_y shape: " + str(y_train.shape))
print("test_x shape: " + str(X_test.shape))
print("test_y shape: " + str(y_test.shape))

y_hat_test = []
y_hat_train = []
# 分解成10个二分类任务。
for i in range(10):
    y_tr = np.zeros(y_train.shape)
    y_te = np.zeros(y_test.shape)
    y_tr[y_train == i] = 1
    y_te[y_test == i] = 1
    d = model(X_train.T, y_tr, X_test.T, y_te, num_iterations=2000, learning_rate=0.5, print_cost=True)
    y_hat_test.append(d["Y_hat_test"])
    y_hat_train.append(d["Y_hat_train"])

    # Print train/test Errors
    print(str(i) + "th classifier train accuracy: {} %".format(100 - np.mean(np.abs(d["Y_prediction_train"] - y_tr)) * 100))
    print(str(i) + "th classifier test accuracy: {} %".format(100 - np.mean(np.abs(d["Y_prediction_test"] - y_te)) * 100))

    """
    # 绘制损失曲线
    plt.figure()
    costs = np.squeeze(d['costs'])
    plt.plot(costs)
    plt.ylabel('cost')
    plt.xlabel('iterations (per hundreds)')
    plt.title(str(i) + "th Classifiers with Learning rate =" + str(d["learning_rate"]))
    plt.show()
    """
    
y_hat_test = np.array(y_hat_test)
y_hat_train = np.array(y_hat_train)

# y_hat_test中存了每个样例在十个分类器正类的置信度。
# 每个样例，选择置信度最高的那个类别，即该样例属于的类别

y_pred_test = np.argmax(y_hat_test, axis=0)
y_pred_train = np.argmax(y_hat_train, axis=0)

score_train = 1.0 * np.sum(y_pred_train == y_train) / m_train
score_test = 1.0 * np.sum(y_pred_test == y_test) / m_test

print(np.sum(y_pred_train == y_train), m_train)
print(np.sum(y_pred_test == y_test), m_test)

print("训练集得分: " + str(score_train))
print("测试集得分: " + str(score_test))
