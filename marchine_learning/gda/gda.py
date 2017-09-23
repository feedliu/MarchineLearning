##################################################
# Copyright: CNIC
# Author: LiuYao
# Date: 2017-9-20
# Description: implements the gaussian discriminant analysis
##################################################

import numpy as np
from numpy.linalg import cholesky
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Ellipse, Circle
from marchine_learning.linear_model import LogisticRegression

def plot_gaussian_and_logistic_boundary(mu_nag, mu_pos, sigma_pos, sigma_nag, theta, x_train, y_train):
    '''
    plot gaussian 3d figure
    '''
    

    R_pos = cholesky(sigma_pos)
    s_pos = np.dot(np.random.randn(100, 2), R_pos) + mu_pos
    p_pos = [compute_gaussian_density(x,mu_pos, sigma_pos).getA()[0][0] for x in s_pos]

    R_nag = cholesky(sigma_nag)
    s_nag = np.dot(np.random.randn(100, 2), R_nag) + mu_nag
    p_nag = [compute_gaussian_density(x,mu_nag, sigma_nag).getA()[0][0] for x in s_nag]

    data_pos = np.column_stack((s_pos, p_pos))
    data_nag = np.column_stack((s_nag, p_nag))
    data = np.vstack((data_pos, data_nag))
    # plt.scatter(data_pos[:, 0], data_pos[:, 1], marker='x')
    # plt.scatter(data_nag[:, 0], data_nag[:, 1], marker='o')

    fig = plt.figure()
    ax = fig.add_subplot(111)
    [m, n] = x_train.shape
    ax.scatter(x_train[:, 0], x_train[:, 1], c=y_train, linewidths=0.1)
    # x1 = np.random.rand(100, 1) * 25
    # x2 = (-theta[2] - x1 * theta[0]) / theta[1]
    # plt.plot(x1, x2)
    ell1 = Ellipse(xy = mu_nag, width = 6, height = 6, angle = 90, facecolor= 'red', alpha=0.3)
    ell2 = Ellipse(xy = mu_pos, width = 6, height = 6, angle = 90, facecolor= 'red', alpha=0.3)
    # cir1 = Circle(xy = (0.0, 0.0), radius=2, alpha=0.5)
    ax.add_patch(ell1)
    ax.add_patch(ell2)
    # ax.add_patch(cir1)
    # x, y = 0, 0
    # ax.plot(x, y, 'ro')
    plt.show()

def compute_gaussian_density(x, mu, sigma):
    '''
    compute the probability of x by gaussian distribution
    Args:
        x: the value of the random variable, maybe a value of n demension
    '''
    return 1.0 / (2 * np.pi * np.linalg.det(sigma) ** 0.5) * \
        np.exp(-0.5 * np.mat((x - mu)) * np.mat(sigma).I * np.mat((x - mu)).T)

def get_gaussian_data():
    mu_pos = np.array([-1, -1])
    sigma_pos = np.array([[1, 0], [0, 1]])
    R_pos = cholesky(sigma_pos)
    s_pos = np.dot(np.random.randn(2000, 2), R_pos) + mu_pos
    p_pos = [compute_gaussian_density(x,mu_pos, sigma_pos).getA()[0][0] for x in s_pos]

    mu_nag = np.array([1, 1])
    sigma_nag = np.array([[1, 0], [0, 1]])
    R_nag = cholesky(sigma_nag)
    s_nag = np.dot(np.random.randn(2000, 2), R_nag) + mu_nag
    p_nag = [compute_gaussian_density(x,mu_nag, sigma_nag).getA()[0][0] for x in s_nag]

    data_pos = np.column_stack((s_pos, np.ones((2000,1)), p_pos))
    data_nag = np.column_stack((s_nag, np.zeros((2000,1)), p_nag))
    data = np.vstack((data_pos, data_nag))
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.plot_trisurf(list(data_pos[:, 0]), list(data_pos[:, 1]), list(data_pos[:, 3]), color=(0,1,0,1))
    # ax.plot_trisurf(list(data_nag[:, 0]), list(data_nag[:, 1]), list(data_nag[:, 3]), color=(0,0,1,1))
    # set the x axis and y axis range
    # ax.axis([-6,6,-10,10])
    # ax.set_zlim(0,1) 
    # plt.show()
    return data

def gaussian_discriminant_analysis(x_train, y_train):
    phi = len(y_train[y_train == 1]) / (len(y_train) * 1.0)
    mu_0 = np.sum(x_train[y_train == 0], axis=0) / len(y_train[y_train == 0])
    mu_1 = np.sum(x_train[y_train == 1], axis=0) / len(y_train[y_train == 1])
    mu = [mu_0 if y_train[i]==0 else mu_1 for i in range(len(y_train))]
    Sigma = np.mat(x_train - mu).T * np.mat(x_train - mu) / len(y_train)
    return phi, mu_0, mu_1, Sigma.getA()

def main():
    data = get_gaussian_data()
    phi, mu_0, mu_1, sigma = gaussian_discriminant_analysis(data[:, 0:2], data[:, 2])
    print phi, mu_0, mu_1, sigma
    # lr = LogisticRegression()
    # theta = lr.train(data[:, 0:2], data[:, 2])
    plot_gaussian_and_logistic_boundary(mu_0, mu_1, sigma, sigma, [1,2,3], data[:, 0:2], data[:, 2])

if __name__ == '__main__':
    main()