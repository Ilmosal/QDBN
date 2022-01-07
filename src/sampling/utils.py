"""
Utility functions go here
"""

import numpy as np

def unpackbits(val, num_bits):
    val = np.array([val])
    valshape = list(val.shape)
    val = val.reshape([-1, 1])
    mask = 2 ** np.arange(num_bits, dtype=val.dtype).reshape([1, num_bits])
    return (val & mask).astype(bool).astype(int).reshape(valshape + [num_bits])

def sample(val):
    return (val > np.random.uniform(0.0, 1.0, val.shape)).astype(float)

def sigmoid(val):
    np.clip(val, a_min = -700, a_max=None, out=val)
    return 1.0 / (1.0 + np.exp(-val))

def l1_between_models(base_model, estimate_model):
    vh_base = np.dot(base_model[0].transpose(), base_model[1]) / len(base_model[0])
    vh_est = np.dot(estimate_model[0].transpose(), estimate_model[1]) / len(estimate_model[0])

    return np.sum(np.abs(vh_base - vh_est))

def get_destination_folder():
    s3_folders = ()
    with open('destination_folder.txt', 'r') as dest_fold_file:
        s3_folders[0] = destination_folder.readline()
        s3_folders[1] = destination_folder.readline()

    return s3_folders

