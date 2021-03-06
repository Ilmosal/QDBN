"""
Module for container object for the mnist data_set
"""

import numpy as np
import cv2

class MnistDataset(object):
    def __init__(self, data_set_path, evaluation_set_path, size_reduction = False):
        self.training_data = np.array([])
        self.evaluation_data = np.array([])
        self.batch_size = 0
        self.batches = 0
        self.data_vector_size = 784
        self.label_vector_size = 10

        tmp_training_data = np.loadtxt(data_set_path, delimiter=',')
        tmp_evaluation_data = np.loadtxt(evaluation_set_path, delimiter=',')

        np.random.shuffle(tmp_training_data)

        if size_reduction:
            self.data_vector_size = 196

            tr_imgs = tmp_training_data[:,:-1]
            ev_imgs = tmp_evaluation_data[:,:-1]
            new_tr_data = np.zeros([len(tmp_training_data), self.data_vector_size + 1])
            new_ev_data = np.zeros([len(tmp_evaluation_data), self.data_vector_size + 1])

            for i in range(len(tmp_training_data)):
                new_tr_data[i] = np.concatenate((np.reshape(cv2.resize(np.reshape(tr_imgs[i], (28, 28)), dsize = (14, 14)), 196), tmp_training_data[i,-1:]))

            for i in range(len(tmp_evaluation_data)):
                new_ev_data[i] = np.concatenate((np.reshape(cv2.resize(np.reshape(ev_imgs[i], (28, 28)), dsize = (14, 14)), 196), tmp_evaluation_data[i,-1:]))

            tmp_training_data = new_tr_data
            tmp_evaluation_data = new_ev_data

        # Form data in a formation where the first are the actual data and last 10 elements are the labels
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

    def get_validation_data(self, batch_size, validation_set):
        return self.training_data[:validation_set * batch_size,:-10]

    def get_validation_labels(self, batch_size, validation_set):
        return self.training_data[:validation_set * batch_size,-10:]

    def get_batches(self, batch_size, include_labels = False, validation_set = 0):
        """
        Function for formatting the batches correctly
        """
        batch_amount = int(len(self.get_training_data()) / batch_size)

        if include_labels:
            data_set = np.copy(self.get_training_data())
        else:
            data_set = np.copy(self.get_training_data_without_labels())

        return np.reshape(data_set, [batch_amount, batch_size, len(data_set[0])])[validation_set:]

class BarsAndStripes():
    """
    Class containing a definition for the bars and stripes dataset
    """
    def __init__(self, img_size, error_rate, tr_samples = 10000, ev_samples = 2000, use_offsets = False, seed = 1000):
        self.training_data = np.array([])
        self.evaluation_data = np.array([])
        self.batch_size = 0
        self.batches = 0

        if isinstance(img_size, list):
            self.data_vector_size = img_size[0]*img_size[1]
            self.image_width = img_size[0]
            self.image_height = img_size[1]
        elif isinstance(img_size, int):
            self.data_vector_size = img_size*img_size
            self.image_width = img_size
            self.image_height = img_size
        else:
            raise Exception("Error trying to generate dataset! img_size not of type list or integer.")

        self.label_vector_size = 2
        self.random_generator = np.random.default_rng(seed)

        self.error_rate = error_rate
        self.use_offsets = use_offsets

        self.training_data = self.gen_n_samples(tr_samples)
        self.evaluation_data = self.gen_n_samples(ev_samples)

    def gen_n_samples(self, n_samples):
        samples = np.zeros([n_samples, self.data_vector_size + self.label_vector_size])

        if self.use_offsets:
            offsets = self.random_generator.integers(0, 2, n_samples)
        else:
            offsets = np.zeros(n_samples)

        # Generate vertical stripes
        for n in range(int(n_samples / 2)):
            for i in range(self.data_vector_size-2):
                if i % 2 == offsets[n]:
                    samples[n, i] = 1.0
                if self.random_generator.random() < self.error_rate:
                    samples[n, i] = np.floor((self.random_generator.random() * 2))

            # Add label
            samples[n,-2] = 1

        # Generate horizontal stripes
        for n in range(int(n_samples / 2), n_samples):
            for i in range(self.data_vector_size - 2):
                if int(i / self.image_width) % 2 == offsets[n]:
                    samples[n, i] = 1.0
                if self.random_generator.random() < self.error_rate:
                    samples[n, i] = int((self.random_generator.random() * 2))

            samples[n,-1] = 1

        self.random_generator.shuffle(samples)

        return samples

    def get_training_data(self):
        return self.training_data

    def get_training_data_without_labels(self):
        return self.training_data[:,:-2]

    def get_training_labels(self):
        return self.training_data[:,-2:]

    def get_evaluation_data(self):
        return self.evaluation_data

    def get_evaluation_data_without_labels(self):
        return self.evaluation_data[:,:-2]

    def get_evaluation_labels(self):
        return self.evaluation_data[:,-2:]

    def get_validation_data(self, batch_size, validation_set):
        return self.training_data[:validation_set * batch_size,:-2]

    def get_validation_labels(self, batch_size, validation_set):
        return self.training_data[:validation_set * batch_size,-2:]

    def get_batches(self, batch_size, include_labels = False, validation_set = 0):
        """
        Function for formatting the batches correctly
        """
        batch_amount = int(len(self.get_training_data()) / batch_size)

        if include_labels:
            data_set = np.copy(self.get_training_data())
        else:
            data_set = np.copy(self.get_training_data_without_labels())

        return np.reshape(data_set, [batch_amount, batch_size, len(data_set[0])])[validation_set:]

