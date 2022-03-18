"""
Class for estimating the model distribution of an RBM with contrastive divergence algorithm.
"""

import numpy as np

from sampling.model import Model
from sampling.utils import sample, sigmoid

class ModelCD(Model):
    """
    Base class for model
    """
    def __init__(self, cd_iter, use_state = True, seed=None):
        super(ModelCD, self).__init__("model_cd")

        self.cd_iter = cd_iter
        self.generator = np.random.default_rng(seed)
        self.dataset = None
        self.use_state = use_state

    def set_model_parameters(self, sampler_parameters):
        """
        Set parameters for contrastive divergence
        """
        self.weights = sampler_parameters['weights']
        self.visible = sampler_parameters['visible']
        self.hidden = sampler_parameters['hidden']
        self.dataset = sampler_parameters['dataset']
        self.label_mode = sampler_parameters['label_mode']

        if self.label_mode == 'active':
            self.label_weights = sampler_parameters['label_weights']
            self.label_biases = sampler_parameters['label_biases']

    def estimate_model(self):
        """
        Estimate the model distribution by cd algorithm.
        """
        if self.dataset is None:
            raise Exception("Dataset not set for the contrastive divergence!")

        # Apply contrastive divergence steps
        vis_state = np.copy(self.dataset[0])
        hid_state = np.copy(self.dataset[1])

        if self.label_mode == 'active':
            lab_state = np.zeros(self.label_biases.shape)
        else:
            lab_state = None

        if not self.use_state:
            vis_state *= 0.0
            vis_state += 0.5

            hid_state = self.activate_hidden(vis_state)

        for i in range(self.cd_iter):
            vis_state = self.activate_visible(hid_state, True)

            if i + 1 == self.cd_iter:
                hid_state = self.activate_hidden(vis_state, lab_state, True)
            else:
                hid_state = self.activate_hidden(vis_state, lab_state)

        return [vis_state, hid_state]

    def activate_hidden(self, values, label_values = None, exact = False):
        """
        Return sampled hidden units for visible values
        """
        labels_influence = np.zeros(self.hidden.shape)

        if self.label_mode == 'active':
            labels_influence = np.dot(label_values, self.label_weights)

        if exact:
            return sigmoid(np.dot(values, self.weights) + self.hidden + labels_influence)
        else:
            return sample(sigmoid(np.dot(values, self.weights) + self.hidden + labels_influence))

    def activate_visible(self, values, exact = False):
        """
        Return sampled visible units for hidden values
        """
        if exact:
            return sigmoid(np.dot(values, self.weights.transpose()) + self.visible)
        else:
            return sample(sigmoid(np.dot(values, self.weights.transpose()) + self.visible))

    def activate_labels(self, values, exact = False):
        """
        Return sampled label units from hidden values
        """
        if exact:
            return softmax(np.dot(values, self.label_weights.transpose()) + self.label_weights)
        else:
            return sample(softmax(np.dot(values, self.label_weights.transpose()) + self.label_weights))

