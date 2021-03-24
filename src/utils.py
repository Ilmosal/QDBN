"Random utility functions"

import sys
import copy
import numpy as np
import matplotlib.pyplot as plt

def compute_weight_reg(weights, reg_con):
    """
    Function for computing and returning the weight regularization matrix for the weights
    """
    # Weight using the L2 regularisation
    weight_mat_copy = copy.deepcopy(weights)

    reg_matrix = 2 * reg_con * weight_mat_copy
    
    return reg_matrix

def sigmoid(val):
    np.clip(val, a_min = -700, a_max=None, out=val)
    return 1.0/(1.0 + np.exp(-val))

def sigmoid_derivative(val):
    return sigmoid(val) * (1 - sigmoid(val))

def cost_derivative(output, labels):
    return (output - labels)

def softmax(val):
    val -= np.max(val, axis=1, keepdims=True)
    sum_val = np.sum(np.exp(val), axis=1, keepdims=True)
    return np.exp(val) / sum_val

def sample(val):
    return (val > np.random.uniform(0.0, 1.0, val.shape)).astype(float)

def activate_sigmoid(val):
    return sample(sigmoid(val))

def activate_softmax(val, s_sum):
    return sample(softmax(val, s_sum))

def loadMNIST(path_to_dir):
    training_data = np.loadtxt(path_to_dir + "mnist_train.csv", delimiter=',')
    test_data = np.loadtxt(path_to_dir + "mnist_test.csv", delimiter=',')

    return training_data, test_data

def plot_letter(letter_data):
    """
    Plot a single letter using matplotlib
    """
    fig = plt.figure
    plt.imshow(np.reshape(letter_data, [28, 28]), cmap='gray')
    plt.show()

