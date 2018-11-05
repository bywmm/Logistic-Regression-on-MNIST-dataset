# -*- coding: utf-8 -*-


import numpy as np
import pandas as pd


def sigmoid(z):
    
    s = 1.0 / (1.0 + np.exp(-z))
    
    return s


def model(X_train, Y_train, X_test, Y_test, num_iterations=2000, learning_rate=0.5, print_cost=False):
    """
    逻辑回归模型
    
    参数：
    X_train -- np数组(num_px * num_px, m_train)
    Y_train -- np数组(1, m_train)
    X_test -- np数组(num_px * num_px, m_test)
    Y_test -- np数组(1, m_test)
    num_iterations -- 超参数 迭代次数
    learning_rate -- 超参数 学习率
    print_cost -- True：每100次迭代打印cost
    
    """

    # 训练集样本数
    m_train = X_train.shape[1]
    # 测试集样本数
    m_test = X_test.shape[1]

    # 初始化w和b为0
    # w 权重, np数组(num_px * num_px, 1)
    w = np.zeros((X_train.shape[0], 1))
    b = 0

    # 新建一个数组,实时记录损失函数（我们的优化目标）,后面可以画出来看看梯度下降的效果
    costs = []
    # 进行num_iterations次迭代,每次迭代算一次梯度下降
    for i in range(num_iterations):
    
        # 首先求出线性部分和激活函数
        A = sigmoid(np.dot(w.T, X_train) + b)
        # 计算损失函数
        cost = -1.0 / m_train * np.sum(Y_train * np.log(A) + (1 - Y_train) * np.log(1 - A))  
        cost = np.squeeze(cost)
    
        # 求梯度
        dw = 1.0 / m_train * np.dot(X_train, (A - Y_train).T)
        db = 1.0 / m_train * np.sum(A - Y_train)
        
        # 梯度下降
        w = w - learning_rate * dw
        b = b - learning_rate * db

        # 记录一下损失
        if i % 100 == 0:
            costs.append(cost)
        
        # 每一百次迭代打印一次损失
        
        if print_cost and i % 100 == 0:
            print("Cost after iteration %i: %f" % (i, cost))

    w = w.reshape(-1, 1)
    
    # 训练集上的预测置信度predict y_hat
    y_hat_train = sigmoid(np.dot(w.T, X_train) + b)
    # 测试集上的预测置信度 y_hat
    y_hat_test = sigmoid(np.dot(w.T, X_test) + b)
    # 训练集上的预测类别
    y_prediction_train = np.zeros((1, m_train))
    y_prediction_train[y_hat_train > 0.5] = 1
    # 测试集上的预测类别
    y_prediction_test = np.zeros((1, m_test))
    y_prediction_test[y_hat_test > 0.5] = 1

    d = {"costs": costs,
         "Y_prediction_test": y_prediction_test,
         "Y_prediction_train": y_prediction_train,
         "Y_hat_test": y_hat_test,
         "Y_hat_train": y_hat_train,
         "w": w,
         "b": b,
         "learning_rate" : learning_rate,
         "num_iterations": num_iterations}
    
    return d
