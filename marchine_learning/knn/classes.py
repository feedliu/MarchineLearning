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

        #1. get the distances between each sample in train samples and each sample in test samples,
        #the distances matrix's shape is (m_test, m_train).
        dists = self.__distance__(x_train, x_test)
        #2. sort the distances by row, and get the sort index
        sort_idx = np.argsort(dists, axis=1)
        #3. get the x index and y index, which is top k distance sample index
        x_idx = np.tile(np.mat(range(m_test)).T, [1, self.k])
        y_idx = sort_idx[:, 0 : self.k]
        #4. get the top k distance labels, and the matrix's shape is (m_test, k)
        labels = np.tile(y_train.T, [m_test, 1])
        p_labels = labels[x_idx, y_idx]
        #5. get the mode of each row, which means the most labels
        y_predict = np.mat(mode(p_labels, axis=1)[0])
        return y_predict
    
    def __distance__(self, x_train, x_test):
        '''
        force compute to get the distance between each sample in train samples and each sample in test samples
        '''
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

    def create_kd_tree(self, datalist):
        '''
        create KD tree
        Args:
            data: data list
        '''
        root = KDNode()
        self.build_tree(root, datalist)
        self.kd_tree = root
        return root

    def build_tree(self, parent, datalist):
        '''
        recursive build tree function
        Args:
            parent: parent node
        '''
        m = datalist.shape[0]
        #if the length of data is equal to 1, the node is a leaf node
        if m == 1:
            parent.data = datalist
            return
        
        #compute the best split demension by the variance of each demension of the data
        demension = np.argmax(np.var(datalist, axis=0))
        #sort the data by the chosen demension
        sorted_index = np.argsort(datalist[:, demension], axis=0)
        #get the index of the middle value in the datalist
        middle = m / 2
        #get the left data
        l_data = datalist[np.squeeze(sorted_index[0 : middle].getA()), :]
        #get the right data
        r_data = datalist[np.squeeze(sorted_index[middle + 1 : ].getA()), :]

        #assign the property of the parent node
        parent.data = datalist[np.squeeze(sorted_index[middle, :].getA())]
        parent.demension = demension
        parent.split_value = datalist[np.squeeze(sorted_index[middle, :].getA()), demension]

        #recursive build the child node if the length of rest data is not equal to zero
        if len(l_data) != 0:
            l_node = KDNode()
            parent.left = l_node
            self.build_tree(l_node, l_data)
        
        if len(r_data) != 0:
            r_node = KDNode()
            parent.right = r_node
            self.build_tree(r_node, r_data)

    def __distance_by_kd_tree__(self, x_test):
        '''
        get nearest neighbors matrix by kd_tree search
        '''
        m = x_test.shape[0]
        dists = np.zeros((m, 1))
        count = 0
        for x in x_test:
            dists[count] = self.__find_neighbor__(x, self.kd_tree)
            count += 1
        return np.mat(dists)
            
    
    def __find_neighbor__(self, x, node):
        '''
        recursive find the neighbor of x in kd-tree
        Args:
            the root node of current child tree
        
        steps:
            1. if the current is leaf node, return the data in the node as the nearest neighbor
            2. if the value of x is less than the split value, take the neighbor of left child
               tree as nearest neighbor. And then check if another child tree has the more nearest
               neighbor;
               if the value of x is more than the split value, do it as like mentioned above;
            3. check if the current node and x has more nearest distance
        '''
        
        if node.demension == None: 
            return node.data
        
        if (x[0, node.demension] <= node.split_value) and node.left:
            neighbor = self.__find_neighbor__(x, node.left)
            if node.right \
                and (np.abs(x[0, node.demension] - node.split_value) < self.__euclidean_distance__(x, neighbor)) \
                and (self.__euclidean_distance__(self.__find_neighbor__(x, node.right), x) < self.__euclidean_distance__(x, neighbor)):
                    neighbor = self.__find_neighbor__(x, node.right)
        elif (x[0, node.demension] > node.split_value) and node.right:
            neighbor = self.__find_neighbor__(x, node.right)
            if node.left \
                and (np.abs(x[0, node.demension] - node.split_value) < self.__euclidean_distance__(x, neighbor)) \
                and (self.__euclidean_distance__(self.__find_neighbor__(x, node.left), x) < self.__euclidean_distance__(x, neighbor)):
                    neighbor = self.__find_neighbor__(x, node.left)
        else:
            # this happens as like:
            # x = 6, node = 5
            #         5
            #        /
            #       4
            neighbor = node.data

        if self.__euclidean_distance__(x, node.data) < self.__euclidean_distance__(x, neighbor):
            neighbor = node.data
        return neighbor

    def __euclidean_distance__(self, x1, x2):
        '''
        compute the euclidean distance
        '''
        return np.sum(np.multiply(x1 - x2, x1 - x2))

class KDNode:
    def __init__(self, data=None, demension=None, split_value=None, left=None, right=None):
        self.data = data
        self.demension = demension
        self.split_value = split_value
        self.left = left
        self.right = right

def main():
    '''
    KNN test unit
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

    

def test_build_tree():
    '''
    test building the kd tree
    '''
    datalist = np.mat([[3, 1, 4],
                       [2, 3, 7],
                       [2, 1, 3],
                       [2, 4, 5],
                       [1, 4, 4],
                       [0, 5, 7],
                       [6, 1, 4],
                       [4, 3, 4],
                       [5, 2, 5],
                       [4, 0, 6],
                       [7, 1, 6]])
    knn = KNN()
    tree = knn.create_kd_tree(datalist)
    res = knn.__find_neighbor__(np.mat([[3,1,5]]), tree)
    print res

if __name__ == '__main__':
    main()
