#encoding=utf-8

'''
implement the knn algorithm
'''

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from scipy.stats import mode
import matplotlib.pyplot as plt

class KNN:
    
    def __init__(self):
        pass

    def predict(self, x_train, y_train, x_test, k=3):
        self.k = k
        m_train = x_train.shape[0]
        m_test = x_test.shape[0]
        x_train = np.mat(x_train)
        y_train = np.mat(y_train)
        x_test = np.mat(x_test)

        dists = self.__distance__(x_train, x_test)
        sort_idx = np.argsort(dists, axis=1)
        x_idx = np.tile(np.mat(range(m_test)).T, [1, self.k])
        y_idx = sort_idx[:, 0 : self.k]
        labels = np.tile(y_train.T, [m_test, 1])
        # p_labels = np.reshape(labels[sort_idx < self.k], (m_test, self.k))
        p_labels = labels[x_idx, y_idx]
        y_predict = np.mat(mode(p_labels, axis=1)[0])
        return y_predict
    
    def __distance__(self, x_train, x_test):
        m_train = x_train.shape[0]
        m_test = x_test.shape[0]
        dists = np.zeros((m_test, m_train))
        count = 0
        for test in x_test:
            test =  np.tile(test, [m_train, 1])
            distance = np.sum(np.multiply(x_train - test, x_train - test), axis=1)
            dists[count] = distance.T
            count += 1
        return dists

def main():
    '''
    test unit
    '''

    #1. load data
    print "1. loading data..."
    data = pd.read_csv('/home/LiuYao/Documents/MarchineLearning/multi_data.csv')
    data['label'] = data['label'] + 1
    x_train, x_test, y_train, y_test = train_test_split(
                                                    data.values[:, 0:2], 
                                                    data.values[:, 2], 
                                                    test_size=0.2, 
                                                    random_state=0
                                                    )

    x_train = np.mat(x_train)
    x_test = np.mat(x_test) 
    y_train = np.mat(y_train).T
    y_test = np.mat(y_test).T

    #2. predict
    print '2. predicting...'
    knn = KNN()
    y_predict = knn.predict(x_train, y_train, x_test, k=1)

    #3. show the results
    print '3. show the results...'
    plt.scatter(x_train.getA()[:, 0], x_train.getA()[:, 1], c=y_train.T.getA()[0], marker='o')
    plt.scatter(x_test.getA()[:, 0], x_test.getA()[:, 1], c=y_predict.T.getA()[0], marker='*')
    plt.show()

if __name__ == '__main__':
    main()
