#coding=utf-8
###############################################################
#Copyright: CNIC
#Author: LiuYao
#Date: 2017-9-15
#Description: implements the CART algorithm
###############################################################

import numpy as np

class CART:

    def __init__(self):
        pass

    def load_data(self,file_name):
        return np.loadtxt(file_name)

    def split_data(self, data, feature, value):
        '''
        split the data to two data sets with the value in the special feature demension
        '''
        data1 = data[np.nonzero(data[:, feature] > value)[0], :]
        data2 = data[np.nonzero(data[:, feature] <= value)[0], :]
        return data1, data2

    def reg_leaf(self, data):
        return np.mean(data[:, -1])

    def reg_error(self, data):
        return np.var(data[:, -1]) * np.shape(data)[0]

    def create_tree(self, data, leaf_func, error_func, ops=(1, 4)):
        '''
        create the tree
        '''
        #choose the best split feature and value
        feature, value = self.choose_best_split(data, leaf_func, error_func, ops)
        if feature == None:
            return value
        tree = {}
        tree['feature'] = feature
        tree['value'] = value
        l_data, r_data = self.split_data(data, feature, value)
        #recursive create left child tree and right child tree
        tree['left'] = self.create_tree(l_data, leaf_func, error_func, ops)
        tree['right'] = self.create_tree(r_data, leaf_func, error_func, ops)
        return tree

    def is_tree(self, obj):
        '''
        judge the current node is a tree
        '''
        return (type(obj).__name__ == 'dict')

    def get_mean(self, tree):
        '''
        recursively get the mean of the current tree
        '''
        if self.is_tree(tree['left']):
            tree['left'] = self.get_mean(tree['left'])
        if self.is_tree(tree['right']):
            tree['right'] = self.get_mean(tree['right'])
        return (tree['left'] + tree['right']) / 2.0

    def prune(self, tree, test_data):
        '''
        use the test data to prune the tree
        '''
        if np.shape(test_data)[0] == 0:
            return self.get_mean(tree)
        if self.is_tree(tree['right']) or self.is_tree(tree['left']):
            l_data, r_data = self.split_data(test_data, tree['feature'], tree['value'])
        if self.is_tree(tree['left']) :
            tree['left'] = self.prune(tree['left'], l_data)
        if self.is_tree(tree['right']) :
            tree['right'] = self.prune(tree['right'], r_data)
        if (not self.is_tree(tree['left'])) and (not self.is_tree(tree['right'])):
            l_data, r_data = self.split_data(test_data, tree['feature'], tree['value'])
            error_no_merge = np.sum(np.power(l_data[:, -1] - tree['left'], 2)) + \
                             np.sum(np.power(r_data[:, -1] - tree['right'], 2))
            tree_mean = (tree['left'] + tree['right']) / 2.0
            error_merge = np.sum(np.power(test_data[:, -1] - tree_mean, 2))
            if error_merge < error_no_merge:
                print 'merging'
                return tree_mean
            else: return tree
        else:
            return tree

    def choose_best_split(self, data, leaf_func, error_func, ops=(1, 4)):
        '''
        choose the best split feature and value
        Args:
            data: shape = (m,n), m is the num of samples, n is the num of features
            leaf_func: generate the leaf node function
            error_func: compute the error function
            ops: ops[0] toler error
                 ops[1] toler num
        '''
        #the minimize tolerance error
        toler_error = ops[0]
        #the minimize tolerance num of the samples of parent node 
        toler_num = ops[1]
        #if the class of the dataset is unique, return the data as leaf node
        if len(np.unique((data[:, -1]).getA().flatten())) == 1:
            return None, leaf_func(data)
        [m, n] = np.shape(data)
        #compute the total error of all dataset
        total_error = error_func(data)
        lowest_error = np.inf
        best_split_value = 0
        best_split_feature = 0
        #traversal all features and values of each
        for feature in range(n - 1):
            for value in set(data[:, feature].getA().flatten()):
                l_data, r_data = self.split_data(data, feature, value)
                #if the num of splited dataset is less than tolerance num,
                #then skip the current feature
                if (np.shape(l_data)[0] < toler_num) or (np.shape(r_data)[0] < toler_num):
                    continue
                #compute the sum error of splited two datasets
                error = error_func(l_data) + error_func(r_data)
                #if the sum error is less than lowest error, remember it
                if error < lowest_error:
                    best_split_feature = feature
                    best_split_value = value
                    lowest_error = error
        #if the decreased value is less than tolerance error, 
        #then return the data as leaf node
        if total_error - lowest_error < toler_error:
            return None, leaf_func(data)
        l_data, r_data = self.split_data(data, feature, value)
        #if the num of splited dataset is less than tolerance num,
        #then return the data as leaf node
        if (np.shape(l_data)[0] < toler_num) or (np.shape(r_data)[0] < toler_num):
            return None, leaf_func(data)
        return best_split_feature, best_split_value

def main():
    cart = CART()
    train_data = np.mat(cart.load_data('/home/LiuYao/Documents/MarchineLearning/marchine_learning/tree/ex2.txt'))
    test_data = np.mat(cart.load_data('/home/LiuYao/Documents/MarchineLearning/marchine_learning/tree/ex2test.txt'))
    tree = cart.create_tree(train_data, cart.reg_leaf, cart.reg_error, ops=(1, 4))
    print tree
    print cart.prune(tree, test_data)

if __name__ == '__main__':
    main()