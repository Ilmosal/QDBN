import sys
import logging
import math
import numpy as np
import utils
import copy
import json
import time

from softmax import Softmax

class RBM(object):
    """
    Base class for the rbm object
    """
    def __init__(self, shape = [1,1], parameters = None, input_included = None):
        if not isinstance(shape,list) and len(shape) != 2:
            logging.error("Shape not an array with two values")

        self.metrics = []
        self.state = np.array([])

        if parameters is not None:
            if type(parameters) is str:
                self.load_parameters(parameters)
            elif type(parameters) is list: 
                self.shape = parameters[0] 
                self.weights = parameters[1]
                self.visible_biases = parameters[2]
                self.hidden_biases = parameters[3]
                self.input_included = parameters[4]

                if self.input_included is not None:
                    self.label_weights = parameters[5]
                    self.label_biases = parameters[6]
        else:
            self.shape = shape

            # Weights initialized from gaussian with stdDev=0.1/sqrt(n_visible*n_hidden) 
            self.weights = (0.1 / np.sqrt(self.shape[0] * self.shape[1])) * np.random.randn(self.shape[0], self.shape[1])

            # Biases initialized at zeros
            self.visible_biases = np.zeros(self.shape[0], dtype=float)
            self.hidden_biases = np.zeros(self.shape[1], dtype=float)

            if input_included is not None:
                self.label_weights = (0.1 / np.sqrt(input_included * self.shape[1])) * np.random.randn(input_included, self.shape[1])
                self.label_biases = np.zeros(input_included, dtype=float)
            else:
                self.label_weights = None
                self.label_biases = None

            self.input_included = input_included

        self.labels_state = np.array([])

    def infer_labels(self, state, exact = False):
        """
        Infer the state of the label units from the hidden units. Only use this function if labels are included with the data
        """
        f = np.dot(state, self.label_weights.transpose()) + self.label_biases

        if exact:
            return utils.softmax(np.dot(state, self.label_weights.transpose()) + self.label_biases)
        else:
            return utils.activate_softmax(np.dot(state, self.label_weights.transpose()) + self.label_biases)
    
    def infer_visible(self, state, exact = False):
        """
        Infer the states of the visible units
        """
        if exact:
            return utils.sigmoid(np.dot(state, self.weights.transpose()) + self.visible_biases)
        else: 
            return utils.activate_sigmoid(np.dot(state, self.weights.transpose()) + self.visible_biases)
            
    def infer_hidden(self, state, exact = False, labels_state = None):
        """
        Infer the states of the hidden units
        """
        if self.input_included:
            if exact:
                return utils.sigmoid(np.dot(state, self.weights) + self.hidden_biases) + utils.softmax(np.dot(labels_state, self.label_weights))
            else:
                return utils.activate_sigmoid(np.dot(state, self.weights) + self.hidden_biases + utils.softmax(np.dot(labels_state, self.label_weights)))
        else:
            if exact:
                return utils.sigmoid(np.dot(state, self.weights) + self.hidden_biases)
            else:
                return utils.activate_sigmoid(np.dot(state, self.weights) + self.hidden_biases)

    def train(self, batches, learning_rate, epochs, cd_iter=1, momentum = 0, regularization_constant = 0.0, max_size = -1, partial_groups = False):
        """
        Train RBM with data.
        """
        #logging.info("Starting the training process")

        batch_size = len(batches[0])
        dw = None
        ldw = None
        scaling = 0.0

        # States of the visible and hidden units initialized randomly for all the batches
        state = [np.random.randint(0, 2, (batch_size, self.shape[0])).astype(float), np.random.randint(0, 2, (batch_size, self.shape[1])).astype(float)]
        
        for e in range(epochs):
            #logging.info("Epoch n. {0}\n----------".format(e+1))
            for batch in batches:
                # Clamp visible to data. Separate labels from batch if included
                if self.input_included is not None:
                    state[0] = copy.deepcopy(batch[:,:-self.input_included])
                    label_state = copy.deepcopy(batch[:,-self.input_included:])
                else:
                    state[0] = copy.deepcopy(batch)

                # Create dropoff weight matrix. This assumes that max_size is some multiple of the length of both state arrays. 
                # The length of each element will be the scaling of that parameter
                dropoff_matrix = np.copy(self.weights)

                if max_size != -1: # Assume layers of equal sizes
                    max_divide = max(math.floor(len(self.hidden_biases) / max_size), math.floor(len(self.visible_biases) / max_size))

                    h_ids = np.arange(len(self.hidden_biases))
                    v_ids = np.arange(len(self.visible_biases))
                    np.random.shuffle(h_ids)
                    np.random.shuffle(v_ids)

                    dropoff_matrix *= 0.0
                    scaling = max_size / len(self.hidden_biases)
                    
                    # Initialize all the full groups
                    for i in range(max_divide):
                        for v_id in v_ids[i*max_size:(i+1)*max_size]:
                            for h_id in h_ids[i*max_size:(i+1)*max_size]:
                                dropoff_matrix[v_id][h_id] = 1.0
                else:
                    scaling = 1.0
                    dropoff_matrix *= 0.0
                    dropoff_matrix += 1.0

                # Create the dropoff weight matrix
                dropoff_weights = dropoff_matrix * self.weights

                # Infer the hidden units from the clamped data
                if self.input_included is not None:
                    label_influence = np.dot(label_state, self.label_weights)
                    state[1] = utils.sigmoid(np.dot(state[0], dropoff_weights) + self.hidden_biases + label_influence)
                    label_copy = copy.deepcopy(label_state)
                else:
                    state[1] = utils.sigmoid(np.dot(state[0], dropoff_weights) + self.hidden_biases)

                state_copy = copy.deepcopy(state)

                # Apply contrastive divergence
                for i in range(cd_iter):
                    state[0] = utils.activate_sigmoid(np.dot(state[1], dropoff_weights.transpose()) + self.visible_biases)
                    if self.input_included is not None:
                        state[1] = utils.activate_sigmoid(np.dot(state[0], dropoff_weights) + self.hidden_biases + label_influence)
                    else:
                        state[1] = utils.activate_sigmoid(np.dot(state[0], dropoff_weights) + self.hidden_biases)

                if self.input_included is not None:
                    label_state = utils.softmax(np.dot(state[1], self.label_weights.transpose()) + self.label_biases)

                #update weights and biases
                dw_tmp = np.dot(state_copy[0].transpose(), state_copy[1]) - np.dot(state[0].transpose(), state[1])
                dw_tmp *= dropoff_matrix
                dw_tmp -= utils.compute_weight_reg(self.weights, regularization_constant)

                if dw is not None:
                    dw_tmp = momentum * dw + (1 - momentum) * dw_tmp

                self.weights += (learning_rate / batch_size) * dw_tmp
                dw = dw_tmp

                self.visible_biases += (learning_rate / batch_size) * np.sum(state[0] - state_copy[0], axis=0)
                self.hidden_biases += (learning_rate / batch_size) * np.sum(state[1] - state_copy[1], axis=0)

                if self.input_included is not None:
                    l_reg_matrix = utils.compute_weight_reg(self.label_weights, regularization_constant)
                    ldw_tmp = (learning_rate / batch_size) * (np.dot(label_copy.transpose(), state_copy[1]) - np.dot(label_state.transpose(), state[1]) - l_reg_matrix)

                    if ldw is not None:
                        self.label_weights += momentum * ldw + (1 - momentum) * ldw_tmp
                    else:
                        self.label_weights += ldw_tmp
                    ldw = ldw_tmp

                    self.label_biases += (learning_rate / batch_size) * np.sum(label_state - label_copy, axis=0)
        
        # Scale the weights so that the expected input to units works as expected
        self.weights *= scaling

    def sample(self, input_value = None, n_samples = 1, cycles = 10):
        """
        Sample data from the network
        """
        random_sample = [np.full([n_samples, self.shape[0]], 0.5), np.full([n_samples, self.shape[1]], 0.5)]
        label_sample = np.zeros([n_samples, self.input_included])

        if self.input_included is not None:
            if input_value is None:
                input_value = np.random.randint(0, 10)
            label_sample[:,input_value] = 1.0

        last = False
        for i in range(cycles):
            if self.input_included is not None:
                random_sample[1] = self.infer_hidden(random_sample[0], labels_state=label_sample)
            else:
                random_sample[1] = self.infer_hidden(random_sample[0])

            if i + 1 == cycles:
                last = True

            random_sample[0] = self.infer_visible(random_sample[1], last)

        return random_sample[0]

    def classify(self, data, cycles = 1):
        """
        Classify data using the input weights
        """
        if self.input_included is None:
            logging.error("Input hasn't been included in the data. Classification is impossible")
            return -1

        class_sample = [np.zeros([len(data), self.shape[0]]).astype(float), np.zeros([len(data), self.shape[1]]).astype(float)]
        label_sample = np.full([len(data), self.input_included], 0.1)
        class_sample[0] = data

        for i in range(cycles):
            class_sample[1] = utils.activate_sigmoid(np.dot(class_sample[0], self.weights) + self.hidden_biases + utils.softmax(np.dot(label_sample, self.label_weights)))
            label_sample = utils.softmax(np.dot(class_sample[1], self.label_weights.transpose()) + self.label_biases)

        return label_sample

    def get_status(self):
        """
        Print out the status of the network
        """
        status_string = "RBM status: "
    
        status_string += "Visible biases"
        status_string += str(self.visible_biases)
        status_string += "\n"

        status_string += "Hidden biases"
        status_string += str(self.hidden_biases)
        status_string += "\n"

        return status_string

    def load_parameters(self, parameter_file_location):
        """
        Function for loading rbm parameters
        """
        logging.info("Loading parameters from file")

        try:
            parameter_file = open(parameter_file_location, 'r')
            parameters = json.load(parameter_file)
            parameter_file.close()

            self.shape = parameters['shape'] 
            self.weights = np.array(parameters['weights'])
            self.visible_biases = np.array(parameters['visible_biases'])
            self.hidden_biases = np.array(parameters['hidden_biases'])
            self.input_included = parameters['input_included']
            self.label_weights = np.array(parameters['label_weights'])
            self.label_biases = np.array(parameters['label_biases'])
        except Exception as e:
            logging.error('Failed to load parameters due to error:')
            logging.error(e)
            return

        logging.info('Parameters successfully read')

    def save_parameters(self, parameter_file):
        """
        Store rbm parameters into a file.
        """
        logging.info("Storing parameters into a file")
        parameters = {
            'shape': self.shape,
            'weights': self.weights.tolist(),
            'visible_biases':self.visible_biases.tolist(),
            'hidden_biases':self.hidden_biases.tolist(),
            'input_included':self.input_included,
            'label_weights':self.label_weights.tolist(),
            'label_biases':self.label_biases.tolist()
        } 

        try:
            parameter_file = open(parameter_file, 'w')
            json.dump(parameters, parameter_file)
            parameter_file.close()
        except Exception as e:
            logging.error('Failed to save parameters due to error:')
            logging.error(e)

