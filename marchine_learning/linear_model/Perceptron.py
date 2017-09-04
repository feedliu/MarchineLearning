#encoding=utf-8
'''
implements the perceptron
'''

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

class Perceptron:
    def __init__(self, alpha=0.1, iterator_num=100):
        self.alpha = alpha
        self.iterator_num = iterator_num

    def train(self, x_train, y_train):
        x_train = np.mat(x_train)
        y_train = np.mat(y_train)
        [m, n] = x_train.shape
        self.theta = np.mat(np.zeros((n, 1)))
        self.b = 0
        self.__stochastic_gradient_decent__(x_train, y_train)

    def __gradient_decent__(self, x_train, y_train):
        x_train = np.mat(x_train)
        y_train = np.mat(y_train)
        for i in xrange(self.iterator_num):
            self.theta = self.theta + self.alpha * x_train.T * y_train
            self.b = self.b + self.alpha * np.sum(y_train)

    def __stochastic_gradient_decent__(self, x_train, y_train):
        x_train = np.mat(x_train)
        y_train = np.mat(y_train)
        [m, n] = x_train.shape
        for i in xrange(self.iterator_num):
            for j in xrange(m):
                self.theta = self.theta + self.alpha * x_train[j].T * y_train[j] 
                self.b = self.b + self.alpha * y_train[j]

def main():
    '''
    test unit
    '''
    print "step 1: load data..."
    data = pd.read_csv('/home/LiuYao/Documents/MarchineLearning/data.csv')
    data = data.ix[0:60, :]
    
    x = np.mat(data[['x', 'y']].values)
    y = np.mat(data['label'].values).T
    y[y == 0] = -1
    print y[y == 1]
    print "positive samples : ", y[y == 1].shape
    print "nagetive samples : ", y[y == -1].shape

    ## step 2: training...
    print "step 2: training..."
    perceptron = Perceptron(alpha=0.1,iterator_num=100)
    perceptron.train(x, y)

    ## step 3: show the decision boundary
    print "step 3: show the decision boundary..."	
    print perceptron.theta
    x_min = np.min(x[:, 0])
    x_max = np.max(x[:, 0])
    y_min = (-perceptron.b - perceptron.theta[0] * x_min) / perceptron.theta[1]
    y_max = (-perceptron.b - perceptron.theta[0] * x_max) / perceptron.theta[1]
    plt.plot([x_min, x_max], [y_min[0,0], y_max[0,0]])
    plt.scatter(x[:, 0].getA(), x[:, 1].getA(), c=y.getA())
    plt.show()

if __name__ == '__main__':
    main()