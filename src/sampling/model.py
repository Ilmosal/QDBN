"""
Root class of the model objects. Used for passing parameters around other models.
"""

import numpy as np

class Model:
    """
    Base class for model
    """
    def __init__(self, model_id = "undefined"):
        self.model_id = model_id

    def estimate_model(self):
        """
        Use sampler to sample the states
        """
        raise Exception("Base class model shouldn't be used")

    def set_model_parameters(self, sampler_parameters):
        """
        Set model parameters.
        """
        raise Exception("Base class model shouldn't be used")

    def get_samples_num(self):
        """
        Get the amount of samples estimate_model returns
        """
        return -1


