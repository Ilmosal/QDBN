"""
Implementation for the deep belief network
"""

import sys
import logging
import json
import copy
import numpy as np
from rbm import RBM
from softmax import Softmax
from utils import sample, sigmoid, sigmoid_derivative, cost_derivative, softmax, compute_weight_reg

class DBN(object):
    """
    Class for a single DBN
    """
    def __init__(self, shape = [1,1], label_shape = 1, parameter_file = None):
        self.shape = shape 
        self.label_shape = label_shape

        self.weights = []
        self.biases = []
        self.gen_weights = []
        self.gen_biases = []

        self.label_weights = []
        self.label_biases = []

        if parameter_file is None:
            for i in range(len(self.shape) - 1):
                # weights initialized with gaussion stDev=0.1/sqrt(n_i * (n_i+1))
                self.weights.append((0.1 / np.sqrt(self.shape[i] * self.shape[i+1])) * np.random.randn(self.shape[i], self.shape[i+1]))
                if i <= len(self.shape) - 2:
                    self.gen_weights.append(np.copy(self.weights[i]))

            for i in range(len(shape)):
                # Biases initialized with zeros
                self.biases.append(np.zeros(self.shape[i], dtype=float))
                if i <= len(self.shape) - 2:
                    self.gen_biases.append(np.zeros(self.shape[i], dtype=float))

            # Normalize label weights with stdev=0.1/((n_label+n_visible)*nhidden). Test later if only n_label works better§
            self.label_weights = (0.1 / np.sqrt((self.shape[-2] + self.label_shape) * self.shape[-1])) * np.random.randn(self.label_shape, self.shape[-1])
            self.label_biases = np.zeros(self.label_shape, dtype=float)
        else:
            self.load_parameters(parameter_file)

    def greedy_pretrain(self, batches, learning_rate, epochs, cd_iter = 1, momentum = 0.0, regularization_constant = 0.0, max_size = -1, labels = False):
        """
        Function for greedily pre-training a stack of RBMs to initialize the weight parameters of the DBN. 
        """
        logging.info('Starting the greedy pretraining process')

        rbms = []

        # create the dataset for the visible layer by inferring the data to the hidden layers of each previous rbm in the stack
        if labels:
            data_set_cpy = np.copy(batches[:,:,:-self.label_shape])
            label_set_cpy = np.copy(batches[:,:,-self.label_shape:])
        else:
            data_set_cpy = np.copy(batches)

        for i in range(len(self.shape) - 1):
            logging.info('Training layer {0}'.format(i+1))

            if i == len(self.shape) - 2 and labels:
                parameters = [
                            [self.shape[i], self.shape[i+1]],
                            self.weights[i],
                            self.biases[i],
                            self.biases[i+1],
                            self.label_shape,
                            self.label_weights,
                            self.label_biases
                ]
            else:
                parameters = [
                            [self.shape[i], self.shape[i+1]],
                            self.weights[i],
                            self.biases[i],
                            self.biases[i+1],
                            None
                ]

            rbm = RBM(parameters=parameters)

            if i != 0:
                # Create new input data array from the last layer of data. Append the label data to the last layer
                if i == len(self.shape) - 2 and labels:
                    new_data_set_array = np.zeros([len(data_set_cpy), len(data_set_cpy[0]), self.shape[i] + self.label_shape])
                    new_data_set_array[:,:,:-self.label_shape] = rbms[-1].infer_hidden(data_set_cpy)
                    new_data_set_array[:,:,-self.label_shape:] = label_set_cpy
                else:
                    new_data_set_array = rbms[-1].infer_hidden(data_set_cpy)

                data_set_cpy = np.zeros([len(data_set_cpy), len(data_set_cpy[0]), self.shape[i]])
                data_set_cpy = new_data_set_array

            # Train the rbm
            rbm.train(
                    data_set_cpy, 
                    learning_rate = learning_rate, 
                    epochs = epochs, 
                    cd_iter = cd_iter, 
                    momentum = momentum, 
                    regularization_constant = regularization_constant, 
                    max_size = max_size)

            rbms.append(rbm)

            if i == len(self.shape) - 2:
                self.biases[i] = np.copy(rbm.visible_biases)
                self.biases[i+1] = np.copy(rbm.hidden_biases)
                self.weights[i] = np.copy(rbm.weights)

                if labels:
                    self.label_biases = np.copy(rbm.label_biases)
                    self.label_weights = np.copy(rbm.label_weights)
            else:
                self.biases[i] = np.copy(rbm.visible_biases)
                self.biases[i+1] = np.copy(rbm.hidden_biases)
                self.weights[i] = np.copy(rbm.weights)
                self.gen_weights[i] = np.copy(rbm.weights)
                self.gen_biases[i] = np.copy(rbm.visible_biases)
                self.gen_biases[i+1] = np.copy(rbm.hidden_biases)

    def wakesleep_algorithm(self, batches, learning_rate, epochs = 1, cycles = 1, momentum = 0.0, regularization_constant = 0.0):
        """
        Function for the wake sleep algorithm presented by Hinton et al
        """
        logging.info("Starting the Wake-Sleep process")

        batch_size = len(batches[0])

        for e in range(epochs):
            dw = [None for i in range(len(self.weights))]
            gdw = [None for i in range(len(self.gen_weights))]
            ldw = None

            logging.info("Epoch n. {0}\n----------".format(e+1))
            for batch in batches:
                # Step 0: Initialize used variables
                batch_cpy= np.copy(batch[:,:-self.label_shape])
                labels_cpy = np.copy(batch[:,-self.label_shape:])
                wake_states = []
                sleep_states = []
                wake_probs = []
                sleep_probs = []
                label_probs = None
                predict_sleep_probs = [None for i in range(len(self.shape) - 1)]
                predict_wake_probs = [None for i in range(len(self.shape) - 2)]
                wake_states.append(np.copy(batch_cpy))

                # Step 1: upwards pass to get wake/positive phase probabilities and sample states
                for i in range(len(self.shape) - 1):
                    if i != len(self.shape) - 2:
                        probs = sigmoid(np.dot(wake_states[i], self.weights[i]) + self.biases[i+1])
                    else:
                        probs = sigmoid(np.dot(wake_states[i], self.weights[i]) + self.biases[i+1] + np.dot(labels_cpy, self.label_weights)) 
                    wake_probs.append(probs)
                    wake_states.append(sample(np.copy(probs)))

                wake_label_statistics = np.reshape(labels_cpy, [batch_size, len(self.label_biases), 1]) * np.reshape(wake_states[-1], [batch_size, 1, len(self.biases[-1])])
                wake_associative_statistics = np.reshape(wake_states[-2], [batch_size, len(self.biases[-2]), 1]) * np.reshape(wake_states[-1], [batch_size, 1, len(self.biases[-1])])

                # Step 2: Gibbs sampling in the top level undirected associative memory
                sleep_probs = copy.deepcopy(wake_probs)
                sleep_states = copy.deepcopy(wake_states)
                
                for i in range(cycles):
                    sleep_probs[-2] = sigmoid(np.dot(sleep_states[-1], self.weights[-1].transpose()) + self.biases[-2])
                    sleep_states[-2] = sample(sleep_probs[-2])
                    label_probs = softmax(np.dot(sleep_states[-1], self.label_weights.transpose()) + self.label_biases)
                    sleep_probs[-1] = sigmoid(np.dot(sleep_states[-2], self.weights[-1]) + self.biases[-1] + np.dot(label_probs, self.label_weights))
                    sleep_states[-1] = sample(sleep_probs[-1])

                sleep_associative_statistics = np.reshape(sleep_states[-2], [batch_size, len(self.biases[-2]), 1]) *  np.reshape(sleep_states[-1], [batch_size, 1, len(self.biases[-1])])
                sleep_label_statistics = np.reshape(label_probs, [batch_size, len(self.label_biases), 1]) * np.reshape(sleep_states[-1], [batch_size, 1, len(self.biases[-1])])

                # Step 3: downwards pass 
                for i in range(len(self.shape) - 2, 1, -1):
                    sleep_probs[i] = sigmoid(np.dot(sleep_states[i+1], self.gen_weights[i].transpose()) + self.gen_biases[i])
                    sleep_states[i] = sample(sleep_probs[i])

                # Step 4: generate predictions
                for i in range(len(self.shape) - 2):
                    predict_sleep_probs[i+1] = sigmoid(np.dot(sleep_states[i], self.weights[i]) + self.biases[i+1])
                    predict_wake_probs[i] = sigmoid(np.dot(wake_states[i+1], self.gen_weights[i].transpose()) + self.gen_biases[i])

                # Step 5: update model parameters
                nabla_gdw = [None for i in range(len(self.gen_weights))]
                nabla_weights = [None for i in range(len(self.weights))]
                nabla_label_weights = None

                # Create weight updates
                for i in range(len(self.shape) - 2):
                    nabla_gdw[i] = np.sum(np.reshape(wake_states[i+1], [batch_size, len(self.gen_biases[i+1]), 1]) * np.reshape(wake_states[i] - predict_wake_probs[i], [batch_size, 1, len(self.gen_biases[i])]), axis = 0).transpose() - compute_weight_reg(self.gen_weights[i], regularization_constant)

                # Update associative parameters
                nabla_label_weights = np.sum(wake_label_statistics - sleep_label_statistics, axis = 0) - compute_weight_reg(self.label_weights, regularization_constant)
                nabla_weights[-1] = np.sum(wake_associative_statistics - sleep_associative_statistics, axis = 0) - compute_weight_reg(self.weights[-1], regularization_constant)

                # Update recognition parameters
                for i in range(len(self.shape) - 2):
                    nabla_weights[i] = np.sum(np.reshape(sleep_states[i], [batch_size, len(self.biases[i]), 1]) * np.reshape((sleep_states[i+1] - predict_sleep_probs[i+1]), [batch_size, 1, len(self.biases[i+1])]), axis = 0) - compute_weight_reg(self.weights[i], regularization_constant)

                for i in range(len(self.shape) - 2):
                    if dw[i] is not None:
                        nabla_weights[i] = momentum * dw[i] + (1 - momentum) * nabla_weights[i]

                    self.weights[i] += (learning_rate / batch_size) * nabla_weights[i]
                    dw[i] = nabla_weights[i]

                    self.biases[i+1] += (learning_rate / batch_size) * np.sum(sleep_states[i+1] - predict_sleep_probs[i+1], axis = 0)

                # Update generative parameters
                for i in range(len(self.shape) - 2):
                    if gdw[i] is not None:
                        nabla_gdw[i] = momentum * gdw[i] + (1 - momentum) * nabla_gdw[i]

                    self.gen_weights[i] += (learning_rate / batch_size) *  nabla_gdw[i]
                    gdw[i] = nabla_gdw[i]

                    self.gen_biases[i] += (learning_rate / batch_size) * np.sum(wake_states[i] - predict_wake_probs[i], axis = 0) 

                # Update associative parameters
                if ldw is not None:
                    nabla_label_weights = momentum * ldw + (1 - momentum) * nabla_label_weights

                self.label_weights += (learning_rate / batch_size) * nabla_label_weights
                ldw = nabla_label_weights

                self.label_biases += (learning_rate / batch_size) * np.sum(labels_cpy - label_probs, axis = 0)

                if dw[-1] is not None:
                    nabla_weights[-1] = momentum * dw[-1] + (1 - momentum) * nabla_weights[-1]

                self.weights[-1] += (learning_rate / batch_size) * nabla_weights[-1]
                dw[-1] = nabla_weights[-1]

                self.biases[-1] += (learning_rate / batch_size) * np.sum(wake_states[-1] - sleep_states[-1], axis = 0)
                self.biases[-2] += (learning_rate / batch_size) * np.sum(wake_states[-2] - sleep_states[-2], axis = 0)

    def finetuning_algorithm(self, batches, learning_rate = 0.01, epochs = 1, momentum = 0.0, regularization_constant = 0.0):
        """
        Function for discriminative learning for the dbn using backpropagation algorithm
        """
        logging.info("Starting the finetuning process")
        for e in range(epochs):
            logging.info("Epoch n. {0}\n----------".format(e+1))
            for batch in batches:
                # Step 0: Initiate helpful variables
                batch_size = len(batch)
                nabla_b = [np.zeros(np.append(np.array([batch_size]), b.shape)) for b in self.biases]
                nabla_b.append(np.zeros(np.append(np.array([batch_size]), self.label_biases.shape)))
                nabla_w = [np.zeros(np.append(np.array([batch_size]), w.shape)) for w in self.weights]
                nabla_w.append(np.zeros(np.append(np.array([batch_size]), self.label_weights.shape)))

                bs = copy.deepcopy(self.biases)
                bs.append(copy.deepcopy(self.label_biases))
                ws = copy.deepcopy(self.weights)
                ws.append(copy.deepcopy(self.label_weights.transpose()))

                batch_cpy= np.copy(batch[:,:-self.label_shape])
                labels_cpy = np.copy(batch[:,-self.label_shape:])

                activation = np.copy(batch_cpy)
                activations = [activation]
                zs = []

                # Step 1: Feedforward pass
                for i in range(len(bs) - 1):
                    z = np.dot(activation, ws[i]) + bs[i+1]
                    zs.append(z)
                    activation = sigmoid(z)
                    activations.append(activation)

                # Step 2: Compute error
                delta = cost_derivative(activations[-1], labels_cpy) * sigmoid_derivative(zs[-1])
                nabla_b[-1] = delta
                nabla_w[-1] = np.reshape(delta, [batch_size, 1, len(delta[0])]) * np.reshape(activations[-2], [batch_size, len(activations[-2][0]), 1])

                # Step 3: Backpropagate the error
                for l in range(2, len(bs)-1):
                    delta = np.dot(delta, ws[-l+1].transpose()) * sigmoid_derivative(zs[-l])
                    nabla_b[-l] = delta
                    nabla_w[-l] = np.reshape(delta, [batch_size, 1, len(delta[0])]) * np.reshape(activations[-l-1], [batch_size, len(activations[-l-1][0]), 1])
            
                # Step 4: Update weights and biases
                for b, nb in zip(bs, nabla_b):
                    b += (learning_rate / batch_size) * np.sum(nb, axis=0)
                for w, nw in zip(ws, nabla_w):
                    w += (learning_rate / batch_size) * np.sum(nw, axis=0)

                self.biases = copy.deepcopy(bs[:-1])
                self.weights = copy.deepcopy(ws[:-1])
                self.label_biases = copy.deepcopy(bs[-1])
                self.label_weights = copy.deepcopy(ws[-1]).transpose()

    def sample(self, input_value = None, nsamples = 1, cycles = 10):
        """
        Create a sample from the dbn. TODO: FIX THIS WHEN EVERYTHING ELSE IS DONE
        """
        samples = np.full([nsamples, self.shape[0]], 0.5)

        # Up pass to initialize the associative memory
        for i in range(len(self.rbms) - 1):
            samples_tmp = self.rbms[i].infer_hidden(samples, True)
            samples = copy.deepcopy(samples_tmp)

        # Attach label to the data arrays
        samples_tmp = np.zeros([nsamples, self.shape[-2] + self.label_shape]).astype(float)
        samples_tmp[:,:-self.label_shape] = samples

        if input_value is None:
            samples_tmp[:,-self.label_shape:] = 0.1
        else:
            samples_tmp[:,-self.label_shape + input_value] = 1.0
        samples = copy.deepcopy(samples_tmp)

        # Sample the associative memory for ncycles
        for i in range(cycles):
            samples_tmp = self.rbms[-1].infer_hidden(samples)
            samples[:,:-self.label_shape] = self.rbms[-1].infer_visible(samples_tmp)

        # Remove labels from data
        samples_tmp = samples[:,:-self.label_shape]
        samples = samples_tmp

        # Downward pass to create the sampled image
        for i in range(len(self.rbms) - 2, 0, -1):
            samples = self.rbms[i].infer_visible(samples)

        samples = self.rbms[0].infer_visible(samples, True)

        return samples

    def classify(self, data, cycles = 1):
        """
        Classify data.
        """
        state = np.copy(data)

        #infer the data for the last layer
        for i in range(len(self.shape) - 1):
            new_state = sigmoid(np.dot(state, self.weights[i]) + self.biases[i+1])
            del state
            state = new_state

        # Classify using softmax
        f = np.dot(state, self.label_weights.transpose())
        f -= np.max(f, axis=1, keepdims=True) 
        sum_f = np.sum(np.exp(f), axis=1, keepdims=True)
        predictions = np.exp(f) / sum_f

        return predictions

    def load_parameters(self, parameter_file_location):
        """
        function for loading parameters to DBN
        """
        logging.info('loading parameters from file')

        try:
            parameter_file = open(parameter_file_location, 'r')
            parameters = json.load(parameter_file)
            parameter_file.close()

            self.shape = parameters['shape'] 
            self.label_shape = parameters['label_shape']

            self.weights = []
            self.biases = []
            self.gen_weights = []
            self.gen_biases = []

            for w in parameters['weights']:
                self.weights.append(np.array(w))

            for b in parameters['biases']:
                self.biases.append(np.array(b))
            
            for w in parameters['weights']:
                self.gen_weights.append(np.array(w))

            for b in parameters['biases']:
                self.gen_biases.append(np.array(b))
            
            self.label_weights = np.array(parameters['label_weights'])
            self.label_biases = np.array(parameters['label_biases'])

        except Exception as e:
            logging.error('Failed to load parameters due to error:')
            logging.error(e)
            return

        logging.info('Parameters successfully read')

    def save_parameters(self, parameter_file_name):
        """
        function for saving the parameters of the DBN
        """
        logging.info("Storing parameters into a file")
        
        weights = []
        biases = []
        gen_weights = []
        gen_biases = []

        for w in self.weights:
            weights.append(w.tolist())

        for b in self.biases:
            biases.append(b.tolist())

        for w in self.gen_weights:
            gen_weights.append(w.tolist())

        for b in self.gen_biases:
            gen_biases.append(b.tolist())

        parameters = {
            'shape': self.shape,
            'label_shape': self.label_shape,
            'weights': weights,
            'biases': biases,
            'gen_weights': gen_weights,
            'gen_biases': gen_biases,
            'label_weights': self.label_weights.tolist(),
            'label_biases': self.label_biases.tolist()
        } 

        try:
            parameter_file = open(parameter_file_name, 'w')
            json.dump(parameters, parameter_file)
            parameter_file.close()
        except Exception as e:
            logging.error('Failed to save parameters due to error:')
            logging.error(e)

