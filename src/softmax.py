"""
Implementation for softmax classifier for the purpose of labeling the dbn
"""

import numpy as np
import logging

class Softmax(object):
    def __init__(self, shape, parameters = None):
        self.shape = shape

        self.state = np.array([])
        self.loss = None

        if parameters is not None:
            self.weights = parameters[0]
            self.biases = parameters[1]
        else:
            self.weights = (0.1 / np.sqrt(self.shape[0] * self.shape[1])) * np.random.randn(self.shape[0], self.shape[1])
            self.biases = np.zeros(self.shape[1], dtype = float)
       
    def train(self, data_set, labels, learning_rate = 0.1, epochs = 1, reg = 1e-3):
        """
        Train softmax function
        """
        for e in range(epochs):
            for i in range(len(data_set)):
                batch = data_set[i]
                num_train = batch.shape[0]
                f = batch.dot(self.weights) + self.biases
                f -= np.max(f, axis = 1, keepdims=True)
                sum_f = np.sum(np.exp(f), axis=1, keepdims=True)
                p = np.exp(f) / sum_f

                dscores = np.copy(p)
                dscores[range(num_train), labels[i]] -= 1 
                dscores /= num_train

                self.weights += - learning_rate * reg * batch.T.dot(dscores)
                self.biases += - learning_rate * np.sum(dscores, axis = 0, keepdims = True)[0]

    def evaluate(self, data_set, labels):
        predict_rate = 0

        f = data_set.dot(self.weights) + self.biases
        f -= np.max(f, axis = 1, keepdims=True)
        sum_f = np.sum(np.exp(f), axis=1, keepdims=True)
        p = np.exp(f) / sum_f

        for i in range(len(data_set)):
            if labels[i] == np.argmax(p[i]):
                predict_rate += 1

        return predict_rate / len(data_set)

    def classify(self, data):
        f = data.dot(self.weights) + self.biases
        f -= np.max(f)
        sum_f = np.sum(np.exp(f))
        p = np.exp(f) / sum_f

        return p
