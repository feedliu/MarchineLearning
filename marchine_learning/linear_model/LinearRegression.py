#encoding=utf-8
###################################################################
#Copyright: CNIC
#Author: LiuYao
#Date: 2017-9-12
#Description: implements the linear regression algorithm
###################################################################

import numpy as np
from numpy.linalg import det
from numpy.linalg import inv
from numpy import mat
from numpy import random
import matplotlib.pyplot as plt
import pandas as pd

class LinearRegression:
    '''
    implements the linear regression algorithm class
    '''

    def __init__(self):
        pass

    def train(self, x_train, y_train):
        '''
        linear regression
        '''
        x_mat = mat(x_train).T
        y_mat = mat(y_train).T
        [m, n] = x_mat.shape
        x_mat = np.hstack((x_mat, mat(np.ones((m, 1)))))
        self.weight = mat(random.rand(n + 1, 1))
        if det(x_mat.T * x_mat) == 0:
            print 'the det of xTx is equal to zero.'
            return
        else:
            self.weight = inv(x_mat.T * x_mat) * x_mat.T * y_mat
        return self.weight

    def locally_weighted_linear_regression(self, test_point, x_train, y_train, k=1.0):
        '''
        locally weighted linear regression
        '''
        x_mat = mat(x_train).T
        [m, n] = x_mat.shape
        x_mat = np.hstack((x_mat, mat(np.ones((m, 1)))))
        y_mat = mat(y_train).T
        test_point_mat = mat(test_point)
        test_point_mat = np.hstack((test_point_mat, mat([[1]])))
        self.weight = mat(np.zeros((n+1, 1)))
        weights = mat(np.eye((m)))
        test_data = np.tile(test_point_mat, [m,1])
        distances = (test_data - x_mat) * (test_data - x_mat).T / (n + 1)
        distances = np.exp(distances / (-2 * k ** 2))
        weights = np.diag(np.diag(distances))
        # weights = distances * weights
        xTx = x_mat.T * (weights * x_mat)
        if det(xTx) == 0.0:
            print 'the det of xTx is equal to zero.'
            return 
        self.weight = xTx.I * x_mat.T * weights * y_mat
        return test_point_mat * self.weight

    def ridge_regression(self, x_train, y_train, lam=0.2):
        '''
        use the l2 norm to regularition
        '''
        x_mat = mat(x_train).T
        [m, n] = np.shape(x_mat)
        x_mat = np.hstack((x_mat, mat(np.ones((m, 1)))))
        y_mat = mat(y_train).T
        self.weight = mat(random.rand(n + 1,1))
        xTx = x_mat.T * x_mat + lam * mat(np.eye(n))
        if det(xTx) == 0.0:
            print "the det of xTx is zero!"
            return
        self.weight = xTx.I * x_mat.T * y_mat
        return self.weight

    def lasso_regression(self, x_train, y_train, eps=0.01, itr_num=100):
        '''
        use the l1 norm to regularition, and optimize the problem with the front step algorithm
        '''
        x_mat = mat(x_train).T
        [m,n] = np.shape(x_mat)
        x_mat = (x_mat - x_mat.mean(axis=0)) / x_mat.std(axis=0)       
        x_mat = np.hstack((x_mat, mat(np.ones((m, 1)))))
        y_mat = mat(y_train).T
        y_mat = (y_mat - y_mat.mean(axis=0)) / y_mat.std(axis=0)
        self.weight = mat(random.rand(n+1, 1))
        best_weight = self.weight.copy()
        for i in range(itr_num):
            print self.weight.T
            lowest_error = np.inf
            for j in range(n + 1):
                for sign in [-1, 1]:
                    weight_copy = self.weight.copy()
                    weight_copy[j] += eps * sign
                    y_predict = x_mat * weight_copy
                    error = np.power(y_mat - y_predict, 2).sum()
                    if error < lowest_error:
                        lowest_error = error
                        best_weight = weight_copy
            self.weight = best_weight
        return self.weight

    def lwlr_predict(self, x_test, x_train, y_train, k=1.0):
        '''
        predict the data with the locally weighted linear regression
        '''
        m = len(x_test)
        y_predict = mat(np.zeros((m, 1)))
        for i in range(m):
            y_predict[i] = self.locally_weighted_linear_regression(x_test[i], x_train, y_train, k)
        return y_predict


    def lr_predict(self, x_test):
        '''
        predict the test data
        '''
        m = len(x_test)
        x_mat = np.hstack((mat(x_test).T, np.ones((m, 1))))
        return x_mat * self.weight

    def plot_lr(self, x_train, y_train):
        '''
        plot the decision boundary
        '''
        x_min = x_train.min()
        x_max = x_train.max()
        y_min = self.weight[0] * x_min + self.weight[1]
        y_max = self.weight[0] * x_max + self.weight[1]
        plt.scatter(x_train, y_train)
        plt.plot([x_min, x_max], [y_min[0,0], y_max[0,0]], '-g')
        plt.show()

    def plot_lwlr(self, x_train, y_train, k=1.0):
        '''
        plot the locally weighted linear regression decision boundary
        '''
        x_min = x_train.min()
        x_max = x_train.max()
        x = np.linspace(x_min, x_max, 1000)
        y = self.lwlr_predict(x, x_train, y_train, k)
        plt.scatter(x_train, y_train)
        plt.plot(x, y.getA()[:, 0], '-g')
        plt.show()
        
    def plot_weight_with_lambda(self, x_train, y_train, lambdas):
        '''
        plot the relation between the weight and the log(lambda)
        '''
        weights = np.zeros((len(lambdas), ))
        for i  in range(len(lambdas)):
            self.ridge_regression(x_train, y_train, lam=lambdas[i])
            weights[i] = self.weight[0]
        plt.plot(np.log(lambdas), weights)
        plt.xlabel('log(lambda)')
        plt.ylabel('weight')
        plt.show()

def main():
    data = pd.read_csv('/home/LiuYao/Documents/MarchineLearning/regression.csv')
    data = data / 30
    x_train = data['x'].values
    y_train = data['y'].values
    regression = LinearRegression()
    # regression.train(x_train, y_train) 
    # y_predict = regression.predict(x_train)
    # regression.plot_lr(x_train, y_train)
    # print '相关系数矩阵：', np.corrcoef(y_train, np.squeeze(y_predict))
    # y_predict = regression.lwlr_predict([[15],[20]], x_train, y_train, k=0.1)
    # regression.plot_lwlr(x_train, y_train, k=0.01)
    # print y_predict
    # regression.ridge_regression(x_train, y_train, lam=1)
    # regression.plot_lr(x_train, y_train)
    # regression.plot_weight_with_lambda(x_train, y_train, np.linspace(1e-6,10,1000))
    regression.lasso_regression(x_train, y_train, itr_num=1000)
    regression.plot_lr(x_train, y_train)

if __name__ == '__main__':
    main()