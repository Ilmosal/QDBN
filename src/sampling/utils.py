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

def sample(val, generator):
    return (val > generator.uniform(0.0, 1.0, val.shape)).astype(float)

def sigmoid(val):
    np.clip(val, a_min = -700, a_max=None, out=val)
    return 1.0 / (1.0 + np.exp(-val))

def l1_between_models(base_model, estimate_model):
    vh_base = np.dot(base_model[0].transpose(), base_model[1]) / len(base_model[0])
    vh_est = np.dot(estimate_model[0].transpose(), estimate_model[1]) / len(estimate_model[0])

#    print(np.sum(estimate_model[0], axis = 0) / len(estimate_model[0]))
#    print(np.sum(estimate_model[1], axis = 0) / len(estimate_model[1]))

#    print(np.sum(base_model[0], axis = 0) / len(base_model[0]))
#    print(np.sum(base_model[1], axis = 0) / len(base_model[1]))

    return np.sum(np.abs(vh_base - vh_est))

def get_destination_folder():
    loc = None
    folder = None

    with open('sampling/destination_folders.txt', 'r') as dest_fold_file:
        loc = dest_fold_file.readline()
        folder = dest_fold_file.readline()

    return (loc[:-1], folder[:-1])
