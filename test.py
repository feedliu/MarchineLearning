#encoding=utf-8
'''
Copyright : CNIC
Author : LiuYao
Date : 2017-8-31
Description : test my algorithm
'''

import pandas as pd
import numpy as np
from marchine_learning.linear_model import LogisticRegression
import matplotlib.pyplot as plt

def load_data():
    '''
    load data
    '''
    data = pd.read_csv('./data.csv')
    x = data[['x', 'y']]
    y = data['label']
    return x, y

def plot(x_train, y_train, theta):
        [m, n] = x_train.shape
        plt.scatter(x_train.values[:, 0], x_train.values[:, 1], c=y_train)
        x1 = np.random.rand(100, 1) * 25
        x2 = (-theta[2] - x1 * theta[0]) / theta[1]
        plt.plot(x1, x2)
        plt.show()

# def train():
#     x, y = load_data()
#     x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
#     lr = LogisticRegression(iterator_num=100)
#     lr.train(x_train.values, y_train.values.T)
#     y_predict = lr.predict(x_test.values)
#     y_predict[y_predict > 0.5] = 1
#     y_predict[y_predict < 0.5] = 0
#     print lr.theta
#     print "accuracy : ", np.sum(y_predict.getA()[0] == y_test.values) / (len(y_test) * 1.0)

def main():
    '''
    program entry
    '''
    x, y = load_data()
    lr = LogisticRegression(iterator_num=5, optimization='sgd')
    lr.train(x.values, y.values.T)
    print lr.theta
    plot(x, y, lr.theta)

if __name__ == '__main__':
    main()