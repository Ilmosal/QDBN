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
    def __init__(self, sampler, shape = [1,1], parameters = None, input_included = None):
        if not isinstance(shape,list) and len(shape) != 2:
            logging.error("Shape not an array with two values")

        self.metrics = []
        self.epochs_trained = 0
        self.state = np.array([])

        self.sampler = sampler

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
            self.weights = np.random.normal(0.0, 0.01, [self.shape[0], self.shape[1]])

            # Biases initialized at zeros
            self.visible_biases = np.zeros(self.shape[0], dtype=float)
            self.hidden_biases = np.zeros(self.shape[1], dtype=float)

            if input_included is not None:
                self.label_weights = np.random.normal(0.0, 0.01, [input_included, self.shape[1]])
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

    def create_dropoff_parameters(self, max_size, v_ids, h_ids):
        """
        Create dropoff paremeters for a rbm with max size set
        """
        # Create dropoff parameter containers
        dropoff_w_matrices = []
        dropoff_h_biases = []
        dropoff_v_biases = []

        drop_w_mask = np.zeros(self.weights.shape)
        drop_v_mask = np.zeros(self.hidden_biases.shape)
        drop_h_mask = np.zeros(self.hidden_biases.shape)

        max_divide = 1

        if max_size != -1: # Assume layers of roughly equal sizes
            max_divide = max(math.floor(self.shape[1] / max_size), math.floor(self.shape[0] / max_size))

            np.random.shuffle(h_ids)
            np.random.shuffle(v_ids)

            # Initialize all the full groups
            for i in range(max_divide):
                d_matrix = np.zeros([max_size, max_size])
                d_h_bias = np.zeros([max_size])
                d_v_bias = np.zeros([max_size])

                # compute dropoff masks
                for v_id in v_ids[i*max_size:(i+1)*max_size]:
                    for h_id in h_ids[i*max_size:(i+1)*max_size]:
                        drop_w_mask[v_id][h_id] = 1.0

                    drop_v_mask[v_id] = 1.0

                for h_id in h_ids[i*max_size:(i+1)*max_size]:
                    drop_h_mask[h_id] = 1.0

                d_j = 0
                d_i = 0

                for v_id in v_ids[i*max_size:(i+1)*max_size]:
                    # Compute the sub weight matrices
                    for h_id in h_ids[i*max_size:(i+1)*max_size]:
                        d_matrix[d_i, d_j] = self.weights[v_id, h_id]
                        d_j += 1
                    # assign the sub visible biases
                    d_v_bias[d_i] = self.visible_biases[v_id]
                    d_j = 0
                    d_i += 1

                # assign the sub hidden biases
                for h_id in h_ids[i*max_size:(i+1)*max_size]:
                    d_h_bias[d_j] = self.hidden_biases[h_id]
                    d_j += 1

                # Append the matrices
                dropoff_w_matrices.append(d_matrix)
                dropoff_v_biases.append(d_v_bias)
                dropoff_h_biases.append(d_h_bias)
        else:
            dropoff_w_matrices.append(np.copy(self.weights))
            dropoff_h_biases.append(np.copy(self.hidden_biases))
            dropoff_v_biases.append(np.copy(self.visible_biases))

            drop_w_mask = np.ones(self.weights.shape)
            drop_h_mask = np.ones(self.hidden_biases.shape)
            drop_v_mask = np.ones(self.visible_biases.shape)

        return dropoff_w_matrices, dropoff_h_biases, dropoff_v_biases, drop_w_mask, drop_h_mask, drop_v_mask, max_divide

    def train(self, batches, learning_rate, epochs, cd_iter=1, momentum = 0, regularization_constant = 0.0, max_size = -1, partial_groups = False):
        """
        Train RBM with data.
        """
        logging.info("Starting the training process")

        batch_size = len(batches[0])
        dw = np.copy(self.weights) * 0.0
        dbv = np.copy(self.visible_biases) * 0.0
        dbh = np.copy(self.hidden_biases) * 0.0

        if self.input_included:
            dbl = np.copy(self.label_biases) * 0.0
            ldw = np.copy(self.label_weights) * 0.0

        # States of the visible and hidden units initialized randomly for all the batches
        state = [np.random.randint(0, 2, (batch_size, self.shape[0])).astype(float), np.random.randint(0, 2, (batch_size, self.shape[1])).astype(float)]

        if max_size != -1:
            scaling = max_size / len(self.hidden_biases)
            self.weights /= scaling
        else:
            scaling = 1.0

        for e in range(epochs):
            logging.info("Epoch n. {0}".format(self.epochs_trained + 1))
            for batch in batches:
                # Clamp visible to data. Separate labels from batch if included
                if self.input_included is not None:
                    state[0] = copy.deepcopy(batch[:,:-self.input_included])
                    label_state = copy.deepcopy(batch[:,-self.input_included:])
                else:
                    state[0] = copy.deepcopy(batch)

                h_ids = np.arange(len(self.hidden_biases))
                v_ids = np.arange(len(self.visible_biases))

                # Create dropoff parameters
                dropoff_params = self.create_dropoff_parameters(max_size, v_ids, h_ids)

                dropoff_w_matrices = dropoff_params[0]
                dropoff_h_biases = dropoff_params[1]
                dropoff_v_biases = dropoff_params[2]
                dropoff_matrix = dropoff_params[3]
                dropoff_h_mask = dropoff_params[4]
                dropoff_v_mask = dropoff_params[5]
                max_divide = dropoff_params[6]

                # Create the dropoff weight matrix and scale up using scaling
                dropoff_weights = dropoff_matrix * self.weights
                dropoff_v_bias = self.visible_biases * dropoff_v_mask
                dropoff_h_bias = self.hidden_biases * dropoff_h_mask
                label_influence = self.hidden_biases * 0.0

                # compute label influence if included
                if self.input_included is not None:
                    label_influence = np.dot(np.zeros(self.input_included) + 0.5, self.label_weights)
                    label_copy = copy.deepcopy(label_state)

                state[1] = utils.sigmoid(np.dot(state[0], dropoff_weights) + dropoff_h_bias + label_influence)
                state_copy = copy.deepcopy(state)
                state[1] = utils.sample(state[1])

                # Make results array for sampled states
                num_samples = self.sampler.get_samples_num()

                if num_samples == -1:
                    num_samples = batch_size

                sampled_state = [np.zeros((num_samples, self.shape[0])), np.zeros((num_samples, self.shape[1]))]

                # Apply contrastive divergence steps
                for i in range(max_divide):
                    tmp_max_size = max_size

                    if max_size == -1:
                        tmp_max_size = self.shape[1]

                    # Add passive label influence to hidden biases
                    if self.input_included is not None:
                        j = 0
                        for h_id in h_ids[i*tmp_max_size:(i+1)*tmp_max_size]:
                            dropoff_h_biases[i][j] += label_influence[h_id]
                            j += 1
                        j = 0

                    sub_state = [np.zeros([batch_size, tmp_max_size]), np.zeros([batch_size, tmp_max_size])]

                    # Create the sub state for the dropoff network
                    for batch_id in range(batch_size):
                        j = 0
                        for v_id in v_ids[i*tmp_max_size:(i+1)*tmp_max_size]:
                            sub_state[0][batch_id, j] = state[0][batch_id, v_id]
                            j += 1

                        j = 0

                        for h_id in h_ids[i*tmp_max_size:(i+1)*tmp_max_size]:
                            sub_state[1][batch_id, j] = state[1][batch_id, h_id]
                            j += 1

                    # Sample the states for the model distribution
                    sub_state = self.sample_model_distribution(
                            sub_state,
                            dropoff_w_matrices[i],
                            dropoff_v_biases[i],
                            dropoff_h_biases[i])

                    # Get the state back from the sub state
                    for batch_id in range(num_samples):
                        j = 0
                        for v_id in v_ids[i*tmp_max_size:(i+1)*tmp_max_size]:
                            sampled_state[0][batch_id, v_id] = sub_state[0][batch_id, j]
                            j += 1

                        j = 0
                        for h_id in h_ids[i*tmp_max_size:(i+1)*tmp_max_size]:
                            sampled_state[1][batch_id, h_id] = sub_state[1][batch_id, j]
                            j += 1

                if self.input_included is not None:
                    label_state = utils.softmax(np.dot(sampled_state[1], self.label_weights.transpose()) + self.label_biases)

                #update weights and biases 
                dw_tmp = np.dot(state_copy[0].transpose(), state_copy[1]) / batch_size - np.dot(sampled_state[0].transpose(), sampled_state[1]) / num_samples
                dw_tmp *= dropoff_matrix
                dw_tmp -= utils.compute_weight_reg(self.weights, regularization_constant)

                dw_tmp = momentum * dw + (1 - momentum) * dw_tmp

                self.weights += dw_tmp * learning_rate
                dw = dw_tmp

                tmp_bv = (np.sum(state_copy[0], axis=0) / batch_size - np.sum(sampled_state[0], axis=0) / num_samples) * dropoff_v_mask
                tmp_bh = (np.sum(state_copy[1], axis=0) / batch_size - np.sum(sampled_state[1], axis=0) / num_samples) * dropoff_h_mask

                tmp_bv = momentum * dbv + (1 - momentum) * tmp_bv
                tmp_bh = momentum * dbh + (1 - momentum) * tmp_bh

                self.visible_biases += tmp_bv * learning_rate
                self.hidden_biases += tmp_bh * learning_rate

                dbv = tmp_bv
                dbh = tmp_bh

                if self.input_included is not None:
                    ldw_tmp = np.dot(label_copy.transpose(), state_copy[1]) / batch_size - np.dot(label_state.transpose(), sampled_state[1]) / num_samples
                    ldw_tmp -= utils.compute_weight_reg(self.label_weights, regularization_constant)
                    ldw_tmp = momentum * ldw + (1 - momentum) * ldw_tmp

                    self.label_weights += ldw_tmp * learning_rate
                    ldw = ldw_tmp

                    tmp_bl = (np.sum(label_copy, axis=0) / batch_size - np.sum(label_state, axis=0) / num_samples)
                    tmp_bl = momentum * dbl + (1 - momentum) * tmp_bl

                    self.label_biases += tmp_bl * learning_rate
                    dbl = tmp_bl

            self.epochs_trained += 1

        # Scale the weights so that the expected input to units works as expected
        self.weights *= scaling

    def sample_model_distribution(self, state, weights, v_biases, h_biases):
        """
        Apply contrastive divergence to approximate the gradient of the cost function
        """
        self.sampler.set_model_parameters(weights, v_biases, h_biases)

        if self.sampler.model_id == 'model_cd':
            self.sampler.set_dataset(state[0])

        return self.sampler.estimate_model()

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

    def evaluate(self, data, labels, cycles):
        """
        Evaluate RBM and returng the predict rate
        """
        predictions = self.classify(data, cycles)
        pr = 0

        for i in range(len(predictions)):
            if np.argmax(predictions[i]) == np.argmax(labels[i]):
                pr += 1

        return pr / len(predictions)

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

