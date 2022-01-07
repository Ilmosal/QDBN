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

    def set_model_parameters(self, weights, visible, hidden):
        """
        Set model parameters.
        """
        self.weights = weights
        self.visible = visible
        self.hidden = hidden

    def get_samples_num(self):
        """
        Get the amount of samples estimate_model returns
        """
        return -1


