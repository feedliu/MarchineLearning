#encoding=utf-8
'''
Copyright : CNIC
Author : LiuYao
Date : 2017-8-31
Description : Define the LogisticRegression class
'''

import numpy as np

class LogisticRegression(object):
    '''
    implement the lr relative functions
    '''

    def __init__(self, alpha=0.1, iterator_num=100, optimization='sgd'):
        '''
        lr parameters init
        Args:
            alpha: the learning rate, default is 0.1.
            iterator_num: the count of iteration, default is 100.
            optimization: the optimization method, such as 'sgd', 'gd', default is 'sgd'.
        '''
        self.alpha = alpha
        self.iterator_num = iterator_num
        self.optimization = optimization

    def train(self, x_train, y_train):
        '''
        lr train function
        Args:
            x_train: the train data, shape = (m, n), m is the count of the samples, 
                     n is the count of the features
            y_train: the train labels, shape = (m, 1), m is the count of the samples
        '''
        m, n = x_train.shape
        x_train = np.mat(x_train)
        self.theta = np.mat(np.random.rand(n + 1, 1))
        x_train = np.hstack((x_train, np.ones((m, 1))))
        y_train = np.mat(np.reshape(y_train, (m, 1)))
        if(self.optimization == 'gd'):
            self.__gradient_decent__(x_train, y_train)
        elif(self.optimization == 'sgd'):
            self.__stochastic_gradient_decent__(x_train, y_train)
        return self.theta

    def sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-x))

    def __gradient_decent__(self, x_train, y_train):
        [m, n] = x_train.shape
        for i in xrange(self.iterator_num):
            print "step : %d" % i
            self.theta = self.theta - 1.0 / m * self.alpha * np.transpose(x_train) * (self.sigmoid(x_train * self.theta) - y_train)
    
    def __stochastic_gradient_decent__(self, x_train, y_train):
        [m, n] = x_train.shape
        for j in xrange(self.iterator_num):
            data_index = range(m)
            print "step : ", j
            for i in xrange(m):
                #动态调整学习率
                self.alpha = 4 / (1.0 + j + i) + 0.01
                #随机选取一个样本，随后将其从dataIndex中删除
                rand_index = int(np.random.uniform(0, len(data_index)))
                error = self.sigmoid(np.dot(x_train[rand_index, :], self.theta)) - y_train[rand_index]
                self.theta = self.theta - np.multiply(self.alpha, np.multiply(error, x_train[rand_index].T))


    def predict(self, x_test):
        '''
        lr predict function
        Args:
            x_test: the test data, shape = (m, 1), m is the count of the test data
        '''
        [m, n] = x_test.shape
        x_test = np.mat(x_test)
        x_test = np.hstack((x_test, np.ones((m, 1))))
        return self.sigmoid(x_test * self.theta)