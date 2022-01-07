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
    def __init__(self, cd_iter, seed=None):
        super(ModelCD, self).__init__("model_cd")

        self.cd_iter = cd_iter
        self.generator = np.random.default_rng(seed)
        self.dataset = None

    def set_dataset(self, dataset):
        """
        Set the base dataset for the initial states of the CD algorithm
        """
        self.dataset = dataset

    def estimate_model(self):
        """
        Estimate the model distribution by cd algorithm.
        """
        if self.dataset is None:
            raise Exception("Dataset not set for the contrastive divergence!")

        vis_state = np.copy(self.dataset)
        hid_state = self.activate_hidden(vis_state)

        for i in range(self.cd_iter):
            vis_state = self.activate_visible(hid_state, True)

            if i + 1 == self.cd_iter:
                hid_state = self.activate_hidden(vis_state, True)
            else:
                hid_state = self.activate_hidden(vis_state)

        return [vis_state, hid_state]

    def activate_hidden(self, values, exact = False):
        """
        Return sampled hidden units for visible values
        """
        if exact:
            return sigmoid(np.dot(values, self.weights) + self.hidden)
        else:
            return sample(sigmoid(np.dot(values, self.weights) + self.hidden))

    def activate_visible(self, values, exact = False):
        """
        Return sampled visible units for hidden values
        """
        if exact:
            return sigmoid(np.dot(values, self.weights.transpose()) + self.visible)
        else:
            return sample(sigmoid(np.dot(values, self.weights.transpose()) + self.visible))
