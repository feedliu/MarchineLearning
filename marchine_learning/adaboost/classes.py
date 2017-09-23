#encoding=utf-8
######################################################################
#Copyright: CNIC
#Author: LiuYao
#Date: 2017-9-11
#Description: implements the adaboost algorithm
######################################################################
'''
implements the adaboost
'''

import numpy as np
import matplotlib.pyplot as plt

class AdaBoost:
    '''
    implements the adaboost classifier
    '''

    def __init__(self):
        pass

    def load_simple_data(self):
        '''
        make a simple data set
        '''
        data = np.mat([[1.0, 2.0],
                    [2.0, 1.1],
                    [1.3, 1.0],
                    [1.0, 1.0],
                    [2.0, 1.0]])
        labels = [1.0, 1.0, -1.0, -1.0, 1.0]
        return data, labels

    def classify_results(self, x_train, demension, thresh, op):
        '''
        get the predict results by the data, thresh, op and the special demension
        Args:
            x_train: train data
            demension: the special demension
            thresh: the spliting value
            op: the operator, including '<=', '>'
        '''
        y_predict = np.ones((x_train.shape[0], 1))
        if op == 'le':
            y_predict[x_train[:, demension] <= thresh] = -1.0
        else:
            y_predict[x_train[:, demension] > thresh] = -1.0
        return y_predict

    def get_basic_classifier(self, x_train, y_train, D):
        '''
        generate basic classifier by the data and the weight of data
        Args:
            x_train: train data
            y_train: train label
            D: the weight of the data
        '''
        x_mat = np.mat(x_train)
        y_mat = np.mat(y_train).T
        D_mat = np.mat(D)
        [m,n] = x_mat.shape
        num_steps = 10.0
        min_error = np.inf
        best_basic_classifier = {}
        best_predict = np.mat(np.zeros((m, 1)))
        #traverse all demensions to find best demension
        for demension in range(n):
            step_length = (x_mat[:, demension].max() - x_mat[:, demension].min()) / num_steps
            #traverse all spliting range in the special demension to find best spliting value
            for step in range(-1, int(num_steps) + 1):
                #determine which op has lower error
                for op in ['le', 'g']:
                    thresh = x_mat[:, demension].min() + step * step_length
                    y_predict = self.classify_results(x_mat, demension, thresh, op)
                    error = np.sum(D_mat[np.mat(y_predict) != y_mat])
                    if error < min_error:
                        min_error = error
                        best_predict = np.mat(y_predict).copy()
                        best_basic_classifier['demension'] = demension
                        best_basic_classifier['thresh'] = thresh
                        best_basic_classifier['op'] = op
        return best_basic_classifier, min_error, best_predict


    def train(self, x_train, y_train, max_itr=50):
        '''
        train function
        '''
        m = len(x_train)
        n = len(x_train[0])
        D = [1.0/m for i in range(m)]
        D = np.mat(D).T
        self.basic_classifier_list = []
        acc_label = np.mat(np.zeros((m, 1)))
        #generate each basic classifier
        for i in range(max_itr):
            #generate basic classifier
            basic_classifier, error, y_predict = self.get_basic_classifier(x_train, y_train, D)
            print 'y_predict:', y_predict.T
            #compute the basic classifier weight
            alpha = 0.5 * np.log((1 - error) / max(error, 1e-16))
            #compute the data weight
            D = np.multiply(D, np.exp(-1 * alpha * np.multiply(np.mat(y_train).T, np.mat(y_predict))))
            D = D / D.sum()
            print 'D:', D.T
            basic_classifier['alpha'] = alpha
            #store the basic classifier
            self.basic_classifier_list.append(basic_classifier)
            #accmulate the predict results
            acc_label += alpha * y_predict
            print 'acc_label', acc_label
            #compute the total error of all basic classifier generated until now
            total_error = np.sum(np.sign(acc_label) != np.mat(y_train).T) / float(m)
            print 'total_error:', total_error

            #if total error equals to the thresh, then stop
            if total_error == 0.0: 
                break
        return self.basic_classifier_list
        
    def predict(self, x_test):
        '''
        adaboost predict function
        '''
        x_mat = np.mat(x_test)
        m = x_mat.shape[0]
        acc_label = np.mat(np.zeros((m, 1)))
        for i in range(len(self.basic_classifier_list)):
            predict = self.classify_results(x_mat, 
                                self.basic_classifier_list[i]['demension'],
                                self.basic_classifier_list[i]['thresh'],
                                self.basic_classifier_list[i]['op'])
            # accmulate the predict results of each basic classifier
            acc_label += self.basic_classifier_list[i]['alpha'] * predict
        print acc_label
        return np.sign(acc_label)

def main():
    adaboost = AdaBoost()
    data, labels = adaboost.load_simple_data()
    adaboost.train(data, labels, max_itr=9)
    print adaboost.predict([[5,5], [0,0]])

if __name__ == '__main__':
    main()