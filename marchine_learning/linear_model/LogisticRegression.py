#encoding=utf-8
'''
Define the LogisticRegression class
'''

import numpy as np

class LogisticRegression(object):
    '''
    implement the lr relative functions
    '''

    def __init__(self, alpha=0.1, iterator_num=100):
        self.alpha = alpha
        self.iterator_num = iterator_num

    def train(self, x_train, y_train):
        '''
        lr train function
        Args:
            x_train: the train data
            y_train: the train labels
        '''
        m, n = x_train.shape
        x_train = np.mat(x_train)
        self.theta = np.mat(np.random.rand(n + 1, 1))
        x_train = np.hstack((x_train, np.ones((m, 1))))
        y_train = np.mat(y_train)
        self.gradient_decent(x_train, y_train)
        return self.theta

    def sigmoid(self, x):
        return 1 / (1 + np.exp(np.mat(x)))

    def gradient_decent(self, x_train, y_train):
        for i in xrange(self.iterator_num):
            print "step : %d" % i
            self.theta = self.theta - self.alpha * np.transpose(x_train) * (self.sigmoid(x_train * self.theta) - y_train)
    
    def predict(self, x_test):
        [m, n] = x_test.shape
        x_test = np.mat(x_test)
        x_test = np.hstack((x_test, np.ones((m, 1))))
        return self.sigmoid(x_test * self.theta)