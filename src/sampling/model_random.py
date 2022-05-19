"""
Model for sampling random noise for testing purposes
"""

import numpy as np

from sampling.model import Model

class ModelRandom(Model):
    """
    Base class for model
    """
    def __init__(self, seed = None):
        super(ModelRandom, self).__init__("model_random")

        self.generator = np.random.default_rng(seed)

    def set_model_parameters(self, sampler_parameters):
        """
        Set parameters for contrastive divergence
        """
        self.weights = sampler_parameters['weights']
        self.visible = sampler_parameters['visible']
        self.hidden = sampler_parameters['hidden']
        self.dataset = sampler_parameters['dataset']

    def estimate_model(self):
        """
        Estimate the model distribution by cd algorithm.
        """
        if self.dataset is None:
            raise Exception("Dataset not set for random noise!")

        vis_len = len(self.dataset[0])
        hid_len = len(self.dataset[1])

        vis_state = self.generator.random((len(self.dataset), len(self.visible)))
        hid_state = self.generator.random((len(self.dataset), len(self.hidden)))

        return [vis_state, hid_state]

