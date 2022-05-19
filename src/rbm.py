import sys
import logging
import math
import numpy as np
import utils
import copy
import json
import time

from softmax import Softmax
from sampling.model_cd import ModelCD

class RBM(object):
    """
    Base class for the rbm object
    """
    def __init__(self, sampler = None, shape = [1,1], parameters = None, input_included = None, weight_dist = 0.01, seed = None):
        if not isinstance(shape,list) and len(shape) != 2:
            logging.error("Shape not an array with two values")

        if sampler is None:
            self.sampler = ModelCD(1, seed = seed)
        else:
            self.sampler = sampler

        self.metrics = []
        self.epochs_trained = 0
        self.state = np.array([])

        self.generator = np.random.default_rng(seed)

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

            # Weights initialized from gaussian with stdDev=weight_dist
            self.weights = self.generator.normal(0.0, weight_dist, [self.shape[0], self.shape[1]])

            # Biases initialized at zeros
            self.visible_biases = np.zeros(self.shape[0], dtype=float)
            self.hidden_biases = np.zeros(self.shape[1], dtype=float)

            if input_included is not None:
                self.label_weights = self.generator.normal(0.0, 0.01, [input_included, self.shape[1]])
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
            return utils.activate_softmax(np.dot(state, self.label_weights.transpose()) + self.label_biases, self.generator)

    def infer_visible(self, state, exact = False):
        """
        Infer the states of the visible units
        """
        if exact:
            return utils.sigmoid(np.dot(state, self.weights.transpose()) + self.visible_biases)
        else:
            return utils.activate_sigmoid(np.dot(state, self.weights.transpose()) + self.visible_biases, self.generator)

    def infer_hidden(self, state, exact = False, labels_state = None):
        """
        Infer the states of the hidden units
        """
        if self.input_included:
            if exact:
                return utils.sigmoid(np.dot(state, self.weights) + self.hidden_biases) + utils.softmax(np.dot(labels_state, self.label_weights))
            else:
                return utils.activate_sigmoid(np.dot(state, self.weights) + self.hidden_biases + utils.softmax(np.dot(labels_state, self.label_weights)), self.generator)
        else:
            if exact:
                return utils.sigmoid(np.dot(state, self.weights) + self.hidden_biases)
            else:
                return utils.activate_sigmoid(np.dot(state, self.weights) + self.hidden_biases, self.generator)

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

            self.generator.shuffle(h_ids)
            self.generator.shuffle(v_ids)

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

    def create_sampler_parameters(self, dropoff_params, h_ids, v_ids, max_size, label_mode = "passive", lb_constant = 0.5):
        """
        Create sampler parameters with certain dropoff parameters
        """
        if label_mode not in ['passive', 'active']:
            raise Exception('incorrect label mode passed for sampler parameters: {0}'.format(label_mode))

        sampler_params = {}

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

        # Compute the max size of the sampler
        tmp_max_size = max_size

        if max_size == -1:
            tmp_max_size = self.shape[1]

        # Add passive label influence to hidden biases
        if self.input_included is not None and label_mode == "passive":
            label_influence = np.dot(np.zeros(self.input_included) + lb_constant, self.label_weights)

        if self.sampler.model_id == "model_cd":
            sampler_params['weights'] = dropoff_weights
            sampler_params['visible'] = dropoff_v_bias
            sampler_params['hidden'] = dropoff_h_bias
            sampler_params['label_mode'] = label_mode
            sampler_params['label_influence'] = label_influence

            if label_mode == "active":
                sampler_params['label_weights'] = self.label_weights
                sampler_params['label_biases'] = self.label_biases

        elif self.sampler.model_id == "model_dwave":
            sampler_params['weights'] = dropoff_w_matrices
            sampler_params['visible'] = dropoff_v_biases
            sampler_params['hidden'] = dropoff_h_biases
            sampler_params['h_ids'] = h_ids
            sampler_params['v_ids'] = v_ids
            sampler_params['max_size'] = tmp_max_size
            sampler_params['max_divide'] = max_divide
            sampler_params['label_influence'] = label_influence

        return sampler_params, dropoff_weights, dropoff_h_bias

    def train(self, batches, learning_rate, epochs, momentum = 0, regularization_constant = 0.0, max_size = -1, label_mode = 'passive', partial_groups = False, log_batches = False):
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
        state = [self.generator.integers(0, 2, (batch_size, self.shape[0])).astype(float), self.generator.integers(0, 2, (batch_size, self.shape[1])).astype(float)]

        if max_size != -1:
            scaling = max_size / len(self.hidden_biases)
            self.weights /= scaling
        else:
            scaling = 1.0

        for e in range(epochs):
            logging.info("Epoch n. {0}".format(self.epochs_trained + 1))
            for cur_batch_num, batch in enumerate(batches):
                if log_batches:
                    logging.info("Batch: {0}/{1}".format(cur_batch_num + 1, len(batches)))

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

                # Separating dropoff parameters required for learning
                dropoff_matrix = dropoff_params[3]
                dropoff_h_mask = dropoff_params[4]
                dropoff_v_mask = dropoff_params[5]

                # Create sampler parameters depending on the sampler in question
                sampler_params, dropoff_weights, dropoff_h_bias = self.create_sampler_parameters(dropoff_params, h_ids, v_ids, max_size, label_mode)

                # Create the hidden states 
                state[1] = utils.sigmoid(np.dot(state[0], dropoff_weights) + dropoff_h_bias + np.dot(label_state, self.label_weights))
                state_copy = copy.deepcopy(state)
                state[1] = utils.sample(state[1], self.generator)

                sampler_params['dataset'] = state

                # Make copy of label states
                if self.input_included:
                    label_copy = copy.deepcopy(label_state)

                # Make results array for sampled states
                num_samples = self.sampler.get_samples_num()

                if num_samples == -1:
                    num_samples = batch_size

                # Sample for the model distribution states using the dedicated sampler
                sampled_state = self.sample_model_distribution(state, sampler_params)

                # Get the label states
                if self.input_included is not None:
                    # should this be activate softmax instead? softmax can return quite drastic values
                    label_state = utils.softmax(np.dot(sampled_state[1], self.label_weights.transpose()) + self.label_biases)

                # Update weights and biases 
                dw_tmp = (np.dot(state_copy[0].transpose(), state_copy[1]) / batch_size - np.dot(sampled_state[0].transpose(), sampled_state[1]) / num_samples)
                dw_tmp += utils.compute_weight_reg(self.weights, regularization_constant)

                dw_tmp = momentum * dw + dw_tmp * learning_rate

                self.weights += dw_tmp * dropoff_matrix
                dw = dw_tmp

                tmp_bv = (np.sum(state_copy[0], axis=0) / batch_size - np.sum(sampled_state[0], axis=0) / num_samples)
                tmp_bh = (np.sum(state_copy[1], axis=0) / batch_size - np.sum(sampled_state[1], axis=0) / num_samples)

                tmp_bv = momentum * dbv + tmp_bv * learning_rate
                tmp_bh = momentum * dbh + tmp_bh * learning_rate

                self.visible_biases += tmp_bv * dropoff_v_mask
                self.hidden_biases += tmp_bh * dropoff_h_mask

                dbv = tmp_bv
                dbh = tmp_bh

                if self.input_included is not None:
                    ldw_tmp = np.dot(label_copy.transpose(), state_copy[1]) / batch_size - np.dot(label_state.transpose(), sampled_state[1]) / num_samples
                    ldw_tmp += utils.compute_weight_reg(self.label_weights, regularization_constant)
                    ldw_tmp = momentum * ldw + ldw_tmp * learning_rate

                    self.label_weights += ldw_tmp
                    ldw = ldw_tmp

                    tmp_bl = (np.sum(label_copy, axis=0) / batch_size - np.sum(label_state, axis=0) / num_samples)
                    tmp_bl = momentum * dbl + tmp_bl * learning_rate

                    self.label_biases += tmp_bl
                    dbl = tmp_bl

            self.epochs_trained += 1

        # Scale the weights so that the expected input to units works as expected
        self.weights *= scaling

    def sample_model_distribution(self, state, sampler_parameters):
        """
        Apply contrastive divergence to approximate the gradient of the cost function.
        """
        self.sampler.set_model_parameters(sampler_parameters)

        return self.sampler.estimate_model()

    def sample(self, input_value = None, n_samples = 1, cycles = 10):
        """
        Sample data from the network
        """
        random_sample = [np.full([n_samples, self.shape[0]], 0.5), np.full([n_samples, self.shape[1]], 0.5)]
        label_sample = np.zeros([n_samples, self.input_included])

        if self.input_included is not None:
            if input_value is None:
                input_value = self.generator.integers(0, self.input_included)
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
            class_sample[1] = utils.activate_sigmoid(np.dot(class_sample[0], self.weights) + self.hidden_biases + utils.softmax(np.dot(label_sample, self.label_weights)), self.generator)
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

    def compute_bias_statistics(self):
        """
        Compute statistics about the bias parameters of the rbm
        """
        bias_statistics = {}

        min_val = 99999.0
        max_val = -99999.0
        total_val = 0.0

        for b in np.nditer(self.hidden_biases):
            if b > max_val:
                max_val = b
            if b < min_val:
                min_val = b

            total_val += b

        avg_hidden_bias = total_val / np.size(self.hidden_biases)

        bias_statistics['avg_hb'] = avg_hidden_bias
        bias_statistics['max_hb'] = max_val
        bias_statistics['min_hb'] = min_val
        bias_statistics['hidden_biases'] = np.copy(self.hidden_biases)

        min_val = 99999.0
        max_val = -99999.0
        total_val = 0.0

        for b in np.nditer(self.visible_biases):
            if b > max_val:
                max_val = b
            if b < min_val:
                min_val = b

            total_val += b

        avg_visible_bias = total_val / np.size(self.visible_biases)

        bias_statistics['avg_vb'] = avg_visible_bias
        bias_statistics['max_vb'] = max_val
        bias_statistics['min_vb'] = min_val
        bias_statistics['visible_biases'] = np.copy(self.visible_biases)

        if self.input_included is not None:
            min_val = 99999.0
            max_val = -99999.0
            total_val = 0.0

            for b in np.nditer(self.label_biases):
                if b > max_val:
                    max_val = b
                if b < min_val:
                    min_val = b

                total_val += b

            avg_label_bias = total_val / np.size(self.label_biases)

            bias_statistics['avg_lb'] = avg_label_bias
            bias_statistics['max_lb'] = max_val
            bias_statistics['min_lb'] = min_val
            bias_statistics['label_biases'] = np.copy(self.label_biases)

        return bias_statistics

    def compute_weight_statistics(self):
        """
        Compute statistics about the weight parameters of the rbm
        """
        weight_statistics = {}

        min_val = 99999.0
        max_val = -99999.0
        total_val = 0.0

        for w in np.nditer(self.weights):
            if w > max_val:
                max_val = w
            if w < min_val:
                min_val = w

            total_val += w

        avg_w = total_val / np.size(self.weights)

        weight_statistics['avg_w'] = avg_w
        weight_statistics['max_w'] = max_val
        weight_statistics['min_w'] = min_val
        weight_statistics['weights'] = np.copy(self.weights)

        if self.input_included is not None:
            min_val = 99999.0
            max_val = -99999.0
            total_val = 0.0

            for w in np.nditer(self.label_weights):
                if w > max_val:
                    max_val = w
                if w < min_val:
                    min_val = w

                total_val += w

            avg_lw = total_val / np.size(self.label_weights)

            weight_statistics['avg_lw'] = avg_lw
            weight_statistics['max_lw'] = max_val
            weight_statistics['min_lw'] = min_val
            weight_statistics['label_weights'] = np.copy(self.label_weights)


        return weight_statistics

    def compute_reconstruction_error(self, dataset, labels = None):
        """
        Compute the reconstruction error for a given dataset.
        """
        reconstruction_statistics = {}

        hid_values = self.infer_hidden(dataset, False, labels)
        vis_values = self.infer_visible(hid_values, True)

        reconstruction_error = (dataset - vis_values)**2
        total_error = 0
        min_error = 99999.0
        max_error = 0.0

        for r in reconstruction_error:
            error = np.sum(r)

            if error > max_error:
                max_error = error
            if error < min_error:
                min_error = error
            total_error += error

        avg_error = total_error / dataset.shape[0]

        reconstruction_statistics['reconstruction_error'] = reconstruction_error
        reconstruction_statistics['avg_error'] = avg_error
        reconstruction_statistics['min_error'] = min_error
        reconstruction_statistics['max_error'] = max_error

        return reconstruction_statistics

    def log_statistics(self, dataset = None, labels = None, n_cycles = 5):
        """
        Function for logging statistics about the RBM
        """
        w_stat = self.compute_weight_statistics()
        b_stat = self.compute_bias_statistics()

        logging.info("Weight statistics -  avg: {0} min: {1} max:{2}".format(w_stat['avg_w'], w_stat['min_w'], w_stat['max_w']))

        if self.input_included is not None:
            logging.info("Label weight statistics -  avg: {0} min: {1} max:{2}".format(w_stat['avg_lw'], w_stat['min_lw'], w_stat['max_lw']))

        logging.info("Visible bias statistics -  avg: {0} min: {1} max:{2}".format(b_stat['avg_vb'], b_stat['min_vb'], b_stat['max_vb']))
        logging.info("Hidden bias statistics -  avg: {0} min: {1} max:{2}".format(b_stat['avg_hb'], b_stat['min_hb'], b_stat['max_hb']))

        if self.input_included is not None:
            logging.info("Label bias statistics -  avg: {0} min: {1} max:{2}".format(b_stat['avg_lb'], b_stat['min_lb'], b_stat['max_lb']))

        if dataset is not None:
            r_stat = self.compute_reconstruction_error(dataset, labels)
            logging.info("Reconstruction statistics -  avg: {0} min: {1} max:{2}".format(r_stat['avg_error'], r_stat['min_error'], r_stat['max_error']))

            if self.input_included is not None:
                logging.info("Prediction rate on dataset: {0}".format(self.evaluate(dataset, labels, n_cycles)))

    def store_statistics(self, dataset, file_name):
        """
        Function for collecting statistics about the RBM and storing these into a file
        """
        w_stat = self.compute_weight_statistics()
        b_stat = self.compute_bias_statistics()

        if dataset is not None:
            r_stat = self.compute_reconstruction_error(dataset, labels)
        else:
            r_stat = None

        stat_json = {
                'b_stat':  b_stat,
                'w_stat': w_stat,
                'r_stat': r_stat
        }

        with open(file_name, 'w') as stat_file:
            json.dump(stat_json)

    def load_parameters(self, parameter_file_location):
        """
        Function for loading rbm parameters
        """
        logging.info("Loading parameters from file")

        with open(parameter_file_location, 'r') as p_file:
            parameters = json.load(p_file)

        self.shape = parameters['shape']
        self.weights = np.array(parameters['weights'])
        self.visible_biases = np.array(parameters['visible_biases'])
        self.hidden_biases = np.array(parameters['hidden_biases'])
        self.input_included = parameters['input_included']
        self.label_weights = np.array(parameters['label_weights'])
        self.label_biases = np.array(parameters['label_biases'])

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
            'input_included':self.input_included
        }

        if self.input_included:
            parameters['label_weights'] = self.label_weights.tolist()
            parameters['label_biases'] = self.label_biases.tolist()

        with open(parameter_file, 'w') as p_file:
            json.dump(parameters, p_file)
