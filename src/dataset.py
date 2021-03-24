"""
Module for container object for the mnist data_set
"""
import numpy as np

class MnistDataset(object):
    def __init__(self, data_set_path, evaluation_set_path):
        self.training_data = np.array([])
        self.evaluation_data = np.array([])
        self.batch_size = 0
        self.batches = 0
        self.data_vector_size = 784
        self.label_vector_size = 10

        tmp_training_data = np.loadtxt(data_set_path, delimiter=',')
        tmp_evaluation_data = np.loadtxt(evaluation_set_path, delimiter=',')
       
        # Form data in a formation where the first 784 elements are the actual data and last 10 elements are the labels
        # The training data is float normalized between 0 and 1, and label data is 1 or 0
        self.training_data = np.zeros([len(tmp_training_data), self.data_vector_size + self.label_vector_size])
        self.evaluation_data = np.zeros([len(tmp_evaluation_data), self.data_vector_size + self.label_vector_size])

        self.training_data[:,:-10] = tmp_training_data[:,1:]
        self.evaluation_data[:,:-10] = tmp_evaluation_data[:,1:]

        label_array = np.zeros([len(tmp_training_data), 10])
        label_array[range(len(tmp_training_data)), tmp_training_data[:,0].astype(int)] = 1

        self.training_data[:,-10:] = label_array
        self.training_data[:,:-10] /= 255

        label_array = np.zeros([len(tmp_evaluation_data), 10])
        label_array[range(len(tmp_evaluation_data)), tmp_evaluation_data[:,0].astype(int)] = 1

        self.evaluation_data[:,-10:] = label_array
        self.evaluation_data[:,:-10] /= 255
       
    def get_training_data(self):
        return self.training_data

    def get_training_data_without_labels(self):
        return self.training_data[:,:-10]

    def get_training_labels(self):
        return self.training_data[:,-10:]

    def get_evaluation_data(self):
        return self.evaluation_data

    def get_evaluation_data_without_labels(self):
        return self.evaluation_data[:,:-10]

    def get_evaluation_labels(self):
        return self.evaluation_data[:,-10:]

    def get_batches(self, batch_size, include_labels = False):
        """
        Function for formatting the batches correctly
        """
        batch_amount = int(len(self.get_training_data()) / batch_size)

        if include_labels:  
            data_set = np.copy(self.get_training_data())
        else:
            data_set = np.copy(self.get_training_data_without_labels())

        return np.reshape(data_set, [batch_amount, batch_size, len(data_set[0])])