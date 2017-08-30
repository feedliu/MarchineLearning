#coding=utf-8
'''
this is a example to invoke the LR in sklearn moudle
'''

import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import classification_report

def lr_example(): 
    '''
    this is a example to use lr
    '''

    column_names = ['Sample code number', 'Clump Thickness', 'Uniformity of Cell Size',
                    'Uniformity of Cell Shape', 'Marginal Adhesion', 'Single Epithelial Cell Size',
                    'Bare Nuclei', 'Bland Chromatin', 'Normal Nucleoli', 'Mitoses', 'Class']

    url = ('http://archive.ics.uci.edu/ml/machine-learning-databases/breast'
           '-cancer-wisconsin/breast-cancer-wisconsin.data')
    data = pd.read_csv(url, names=column_names)

    data = data.replace(to_replace='?', value=np.nan)

    data = data.dropna(how='any')

    x_train, x_test, y_train, y_test = train_test_split(data[column_names[1:10]],
                                                        data[column_names[10]],
                                                        test_size=0.25, random_state=2017)

    standard_scaler = StandardScaler()
    x_train = standard_scaler.fit_transform(x_train)
    x_test = standard_scaler.fit_transform(x_test)

    l_regression = LogisticRegression()
    sgdc = SGDClassifier()

    l_regression.fit(x_train, y_train)
    lr_y_predict = l_regression.predict(x_test)
    sgdc.fit(x_train, y_train)
    sgdc_y_predict = sgdc.predict(x_test)

    print "Accuracy of LR Classifier : ", l_regression.score(x_test, y_test)
    print classification_report(y_test, lr_y_predict, target_names=['Benign', 'Malignant'])

    print "Accuracy of SGDC Classifier : ", sgdc.score(x_test, y_test)
    print classification_report(y_test, sgdc_y_predict, target_names=['Benign', 'Malignant'])
    print lr_y_predict, y_test

def main():
    '''
    progress entry
    '''
    lr_example()

if __name__ == '__main__':
    main()
