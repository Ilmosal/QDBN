"Random utility functions"

import sys
import copy
import numpy as np
import matplotlib.pyplot as plt

def compute_weight_reg(weights, reg_con = 0.01):
    """
    Function for computing and returning the weight regularization matrix for the weights
    """
    return - 2 * reg_con * weights

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

def activate_softmax(val):
    return sample(softmax(val))

def plot_letter(letter_data):
    """
    Plot a single letter using matplotlib
    """
    plt.imshow(np.reshape(letter_data, [28, 28]), cmap='gray')
    plt.show()

def evaluate_samplers(sampler, bm_sampler, rbm, dataset):
    """
    Function for evaluating the sampling capabilities of two samplers
    """
    dwave_model_parameters = {
            'dataset': None,
            'weights': [rbm.weights],
            'visible': [rbm.visible_biases],
            'hidden': [rbm.hidden_biases],
            'v_ids': np.arange(rbm.shape[0]),
            'h_ids': np.arange(rbm.shape[1]),
            'max_size': rbm.shape[0],
            'max_divide': 1
    }

    cd_model_parameters = {
            'dataset': [dataset, None],
            'weights': rbm.weights,
            'visible': rbm.visible_biases,
            'hidden': rbm.hidden_biases,
            'v_ids': np.arange(rbm.shape[0]),
            'h_ids': np.arange(rbm.shape[1]),
            'max_size': rbm.shape[0],
            'max_divide': 1
    }

    if sampler.model_id == 'model_dwave':
        sampler.set_model_parameters(dwave_model_parameters)
    else:
        sampler.set_model_parameters(cd_model_parameters)

    if bm_sampler.model_id == 'model_dwave':
        bm_sampler.set_model_parameters(dwave_model_parameters)
    else:
        bm_sampler.set_model_parameters(cd_model_parameters)

    s_results = sampler.estimate_model()
    bm_results = bm_sampler.estimate_model()

    samples_num = sampler.get_samples_num()
    bm_samples_num = bm_sampler.get_samples_num()

    if samples_num == -1:
        samples_num = len(dataset)
    if bm_samples_num == -1:
        bm_samples_num = len(dataset)

    s_model_vh = np.dot(s_results[0].transpose(), s_results[1]) / samples_num
    bm_model_vh = np.dot(bm_results[0].transpose(), bm_results[1]) / bm_samples_num

    print(bm_model_vh)

    return np.sum(np.abs(s_model_vh - bm_model_vh))
