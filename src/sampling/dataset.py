"""
This class contains the definition for the randomly created dataset
for the sampling classes. This dataset should be random but the
same each time it will be ran.
"""

import numpy as np

class Dataset():
    """
    Dataset class
    """
    def __init__(self, generator, size, n_samples = 1000):
        self.data = np.zeros([n_samples, size])

        for i in range(n_samples):
            self.data[i] = generator.integers(0, 2, size)

    def get_data(self):
        return self.data

