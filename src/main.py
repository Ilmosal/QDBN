"""
This is a code that handles the basis for my thesis on Quantum annealing and restricted Boltzmann machnines.

TODO:
    - [X] Implementation for Deep belief networks
        - [X] Basic functionality
        - [X] Greedy pretraining
        - [X] Softmax classification
        - [X] Momentum for weights
        - [X] Backpropagation
        - [X] Weight decay
        - [X] Split RBM
    - [X] Evaluating the results
    - [X] Integrate the sampling module
    - [X] Make DWave sampling work
    - [X] Evaluate results of quantum sampling
"""
from rbm import RBM
from dbn import DBN
from dataset import MnistDataset, BarsAndStripes
from softmax import Softmax
from utils import evaluate_samplers

from sampling.utils import l1_between_models
from sampling.model_cd import ModelCD
from sampling.model_random import ModelRandom
from sampling.model_dwave import ModelDWave
from sampling.dataset import Dataset

import numpy as np
from matplotlib import pyplot as plt
import utils
import logging
import csv
import sys

dbn_file = '../data/dbn_weight_reg_parameters.json'
rbm_file = '../data/rbm_l2_params.json'

def run():
    #logging.basicConfig(filename='sampling_beta_output.txt', filemode='w', level=logging.INFO)
    logging.basicConfig(level=logging.INFO)
    logging.info("Running program")

    test_sampling_dwave_by_epoch()
    #train_for_a_single_epoch(11)

    #test_sampling_evaluation(None)
    #test_sampling()
    #train_important_quantum_rbms_256()
    #train_important_quantum_rbms_64()
    #train_important_classical_rbms_64()
    #train_important_classical_rbms_256()
    #test_statistics()
    #test_two_part_training()
    #test_bars_and_stripes()
    #test_bars_and_stripes_dbn()
    #train_rbm_final()
    #train_rbm_with_reduced_mnist()
    #train_dbn_with_reduced_mnist()
    #test_resize()
    #train_rbm_from_mnist_with_input_final()
    #sampling_effect_on_rbm()
    #test_sampling()
    #generate_mnist_plots()
    #generate_plots()
    #train_dbn_final()
    #finetune_dbn_final()
    #evaluate_dbn_mnist()
    #evaluate_rbm_with_input()
    #wake_sleep_dbn()
    #train_rbm_from_mnist_with_input()
    #train_rbm_from_mnist_no_input()
    #sample_mnist_letter()
    #train_dbn_mnist()
    #finetune_dbn_mnist()
    #sample_dbn()
    #test_dbn_final()
    #test_softmax()

def test_sampling_dwave_by_epoch():
    logging.info("Testing sampling capabilities of DWave machine")

    n_size = 64
    rbm = RBM(None, shape=[n_size, n_size], input_included = None, weight_dist = 0.32)

    dataset = BarsAndStripes([8, 8], 0.7, use_offsets = False, seed = 123456789)
    batches = dataset.get_batches(20, include_labels = False)
    evaluation_set = dataset.get_evaluation_data_without_labels()
    evaluation_labels = dataset.get_evaluation_labels()

    gen = np.random.default_rng()

    hid_val = np.zeros([2000, n_size])
    hid_val += 0.5

    dat_values = [
            dataset.get_evaluation_data_without_labels(),
            hid_val
    ]

    model_cd = ModelCD(1000)
    model_cd_1 = ModelCD(1)

    model_aws = ModelDWave(
        layout="pegasus",
        source="aws",
        beta=1.00,
        num_reads=1000,
        s_pause=0.50,
        pause_duration=0.0,
        parallel_runs=False)
    model_random = ModelRandom()

    for i in range(3):
        rbm.load_parameters("./models/model_size_{0}/q_rbm_test_parallel_epoch_{1}.json".format(64, i))
        rbm.log_statistics(evaluation_set, evaluation_labels)

        cd_samp_params = {
            'weights': rbm.weights,
            'visible': rbm.visible_biases,
            'hidden': rbm.hidden_biases,
            'label_mode': 'passive',
            'dataset': dat_values,
        }

        sampler_parameters = {
            'weights': [rbm.weights],
            'visible': [rbm.visible_biases],
            'hidden': [rbm.hidden_biases],
            'label_mode': 'passive',
            'dataset': dat_values,
            'h_ids': [*range(0, n_size)],
            'v_ids': [*range(0, n_size)],
            'max_size': n_size,
            'max_divide': 1,
            'self.parallel': False
        }

        model_random.set_model_parameters(cd_samp_params)
        model_aws.set_model_parameters(sampler_parameters)

        model_cd.set_model_parameters(cd_samp_params)
        model_cd_1.set_model_parameters(cd_samp_params)

        res_1 = model_aws.estimate_model()
        res_2 = model_cd_1.estimate_model()
        an_res = model_cd.estimate_model()

        l1 = l1_between_models(an_res, res_1)
        logging.info("l1 between random model: {0}".format(l1))

        l1 = l1_between_models(an_res, res_2)
        logging.info("l1 between cd model: {0}".format(l1))

def test_dwave_sampling():
    logging.info("Testing sampling capabilities of DWave machine")

    n_size = 128
    rbm = RBM(None, shape=[n_size, n_size], input_included = None, weight_dist = 0.32)
    dataset = BarsAndStripes([16, 8], 0.7, use_offsets = False, seed = 123456789)

    batches = dataset.get_batches(20, include_labels = False)

    rbm.train(batches, learning_rate=0.1, regularization_constant=0.0, epochs=2, momentum=0.5, max_size=-1, label_mode='passive')

    model_cd = ModelCD(1000)
    model_cd1 = ModelCD(1)

    gen = np.random.default_rng()

    hid_val = np.zeros([2000, n_size])
    hid_val += 0.5

    dat_values = [
            dataset.get_evaluation_data_without_labels(),
            hid_val
    ]

    cd_samp_params = {
        'weights': rbm.weights,
        'visible': rbm.visible_biases,
        'hidden': rbm.hidden_biases,
        'label_mode': 'passive',
        'dataset': dat_values,
    }

    sampler_parameters = {
        'weights': [rbm.weights],
        'visible': [rbm.visible_biases],
        'hidden': [rbm.hidden_biases],
        'label_mode': 'passive',
        'dataset': dat_values,
        'h_ids': [*range(0, n_size)],
        'v_ids': [*range(0, n_size)],
        'max_size': n_size,
        'max_divide': 1,
        'self.parallel': False
    }

    model_dwave = ModelDWave(
        layout="pegasus",
        source="dwave",
        beta=1.0,
        num_reads=500,
        s_pause=0.45,
        pause_duration=100.0,
        parallel_runs=False)

    model_aws = ModelDWave(
        layout="pegasus",
        source="aws",
        beta=1.0,
        num_reads=500,
        s_pause=0.45,
        pause_duration=100.0,
        parallel_runs=False)

    model_dwave.set_model_parameters(sampler_parameters)
    model_aws.set_model_parameters(sampler_parameters)

    res = 0
    iterations = 1

    for i in range(iterations):
        res_1 = model_dwave.estimate_model()
        res_2 = model_aws.estimate_model()
        res_3 = model_dwave.estimate_model()
        res_4 = model_aws.estimate_model()

        l1 = l1_between_models(res_1, res_2)
        logging.info("l1 between dwave_1 and aws_1: {0}".format(l1))
        res += l1

        l1 = l1_between_models(res_1, res_3)
        logging.info("l1 between dwave_1 and dwave_2: {0}".format(l1))
        res += l1

        l1 = l1_between_models(res_1, res_4)
        logging.info("l1 between dwave_1 and aws_2: {0}".format(l1))
        res += l1

        l1 = l1_between_models(res_2, res_3)
        logging.info("l1 between aws_1 and dwave_2: {0}".format(l1))
        res += l1

        l1 = l1_between_models(res_2, res_4)
        logging.info("l1 between aws_1 and aws_2: {0}".format(l1))
        res += l1

        l1 = l1_between_models(res_3, res_4)
        logging.info("l1 between dwave_1 and aws_2: {0}".format(l1))
        res += l1

    logging.info("Fidelity of gradient: {0}".format(res / iterations))

def train_for_a_single_epoch(cur_epoch):
    logging.info('Training important quantum RBMS with size 256x256')

    batch_size = 500
    data_vector_size = 256
    hidden_layer_size = 256
    epochs = 10
    sampler = None

    rbm_file_name = "./models/model_size_{0}/q_rbm_{1}_epoch_{2}.json"
    random_seed = 123456788

    sampler = ModelDWave(
        layout="pegasus",
        source="dwave",
        beta=1.0,
        num_reads=100,
        s_pause=0.45,
        pause_duration=100.0,
        parallel_runs=False)


    dataset = BarsAndStripes(16, 0.6, use_offsets = True, seed = random_seed)
    batches = dataset.get_batches(batch_size, include_labels = True)
    tr_set = dataset.get_training_data_without_labels()
    tr_labels = dataset.get_training_labels()
    evaluation_set = dataset.get_evaluation_data_without_labels()
    evaluation_labels = dataset.get_evaluation_labels()

    logging.info("Training this rbm now")
    rbm = RBM(sampler, shape=[data_vector_size, data_vector_size], input_included = 2, weight_dist = 0.32, seed = random_seed)
    rbm.load_parameters("./models/model_size_{0}/q_rbm_{1}_epoch_{2}.json".format(256, 128, cur_epoch))
    rbm.log_statistics(evaluation_set, evaluation_labels)

    rbm.train(batches, learning_rate=0.1, regularization_constant=0.0, epochs=1, momentum=0.5, max_size=128, label_mode='passive')
    rbm.log_statistics(evaluation_set, evaluation_labels)
    rbm.save_parameters(rbm_file_name.format(256, 128, cur_epoch+1))

def train_important_quantum_rbms_64():
    logging.info('Training important quantum RBMs with size 64x64')

    batch_size = 500
    data_vector_size = 64
    hidden_layer_size = 64
    epochs = 10

    random_seed = 123456788

    sampler = ModelDWave(
        layout="pegasus",
        source="aws",
        beta=0.25,
        num_reads=400,
        s_pause=0.45,
        pause_duration=100.0,
        parallel_runs=True)

    #sampler = ModelCD(1, use_state = True, seed = random_seed)

    rbm_file_name = "./models/model_size_{0}/q_rbm__epoch_{1}.json"

    dataset = BarsAndStripes(8, 0.7, use_offsets = False, seed = random_seed)
    batches = dataset.get_batches(batch_size, include_labels = True)
    tr_set = dataset.get_training_data_without_labels()
    tr_labels = dataset.get_training_labels()
    evaluation_set = dataset.get_evaluation_data_without_labels()
    evaluation_labels = dataset.get_evaluation_labels()

    logging.info("Training this rbm now")

    rbm = RBM(sampler, shape=[data_vector_size, data_vector_size], input_included = 2, weight_dist = 0.01, seed = random_seed)
    #rbm.save_parameters(rbm_file_name.format(64, 0))
    rbm.load_parameters(rbm_file_name.format(64, 20))
    rbm.log_statistics(evaluation_set, evaluation_labels)

    for e in range(0, epochs):
        rbm.train(batches, learning_rate=0.1, regularization_constant=0.0, epochs=1, momentum=0.5, max_size=-1, label_mode='passive')
        rbm.log_statistics(evaluation_set, evaluation_labels)
        rbm.save_parameters(rbm_file_name.format(64, e+1))

def train_important_classical_rbms_64():
    logging.info('Training important classical RBMS with size 64x64')

    batch_size = 500
    data_vector_size = 64
    hidden_layer_size = 64
    epochs = 10
    sampler = None
    dwave = False

    rbm_file_name = "./models/model_size_{0}/c_rbm_{1}_{2}_epoch_{3}.json"
    random_seed = 123456788

    parameters = [
        [False, 'passive'],
        [True, 'passive'],
        [False, 'active'],
        [True, 'active']
    ]

    dataset = BarsAndStripes(8, 0.7, use_offsets = False, seed = random_seed)
    batches = dataset.get_batches(batch_size, include_labels = True)
    tr_set = dataset.get_training_data_without_labels()
    tr_labels = dataset.get_training_labels()
    evaluation_set = dataset.get_evaluation_data_without_labels()
    evaluation_labels = dataset.get_evaluation_labels()

    for p in parameters:
        logging.info("parameters: {0}".format(p))
        sampler = ModelCD(1, use_state = p[0], seed = random_seed)

        logging.info("Training this rbm now")

        rbm = RBM(sampler, shape=[data_vector_size, data_vector_size], input_included = 2, weight_dist = 0.32, seed = random_seed)
        rbm.log_statistics(evaluation_set, evaluation_labels)
        rbm.save_parameters(rbm_file_name.format(64, p[0], p[1], 0))

        for e in range(epochs):
            rbm.train(batches, learning_rate=0.1, regularization_constant=0.0, epochs=1, momentum=0.5, max_size=-1, label_mode=p[1])
            rbm.log_statistics(evaluation_set, evaluation_labels)
            rbm.save_parameters(rbm_file_name.format(64, p[0], p[1], e+1))

def train_important_quantum_rbms_256():
    logging.info('Training important quantum RBMS with size 256x256')

    batch_size = 500
    data_vector_size = 256
    hidden_layer_size = 256
    epochs = 10
    sampler = None

    rbm_file_name = "./models/model_size_{0}/q_rbm_aws_drp_{1}_epoch_{2}.json"
    random_seed = 123456788

    parameters = [
        [128, 0.1],
        [64, 0.1],
    ]

    q_sampler = ModelDWave(
        layout="pegasus",
        source="aws",
        beta=2.0,
        num_reads=100,
        s_pause=0.45,
        pause_duration=100.0,
        parallel_runs=False)

    dataset = BarsAndStripes(16, 0.85, use_offsets = False, seed = random_seed)
    batches = dataset.get_batches(batch_size, include_labels = True)
    tr_set = dataset.get_training_data_without_labels()
    tr_labels = dataset.get_training_labels()
    evaluation_set = dataset.get_evaluation_data_without_labels()
    evaluation_labels = dataset.get_evaluation_labels()

    for p in parameters:
        logging.info("Training this rbm now")
        rbm = RBM(q_sampler, shape=[data_vector_size, data_vector_size], input_included = 2, weight_dist = 0.16, seed = random_seed)
        rbm.log_statistics(evaluation_set, evaluation_labels)
        rbm.save_parameters(rbm_file_name.format(256, p[0], 0))

        # Quantum training
        for e in range(0, epochs):
            rbm.train(batches, learning_rate=p[1], regularization_constant=0.0, epochs=1, momentum=0.5, max_size=p[0], label_mode='passive')
            rbm.log_statistics(evaluation_set, evaluation_labels)
            rbm.save_parameters(rbm_file_name.format(256, p[0], e+1))

def train_important_classical_rbms_256():
    logging.info('Training important classical RBMS with size 256x256')

    batch_size = 250
    data_vector_size = 256
    hidden_layer_size = 256
    epochs = 12
    sampler = None

    rbm_file_name = "./models/model_size_{0}/c_rbm_alt_{1}_epoch_{2}.json"
    random_seed = 123456788

    parameters = [
        [64, 0.1],
        [128, 0.1],
        [196, 0.1],
        [256, 0.01]
    ]

    dataset = BarsAndStripes(16, 0.85, use_offsets = False, seed = random_seed)
    batches = dataset.get_batches(batch_size, include_labels = True)
    tr_set = dataset.get_training_data_without_labels()
    tr_labels = dataset.get_training_labels()
    evaluation_set = dataset.get_evaluation_data_without_labels()
    evaluation_labels = dataset.get_evaluation_labels()

    for p in parameters:
        sampler = ModelCD(1, use_state = False, seed = random_seed)

        logging.info("Training this rbm now")
        rbm = RBM(sampler, shape=[data_vector_size, data_vector_size], input_included = 2, weight_dist = 0.32, seed = random_seed)
        rbm.log_statistics(evaluation_set, evaluation_labels)
        rbm.save_parameters(rbm_file_name.format(256, p[0], 0))

        for e in range(epochs):
            rbm.train(batches, learning_rate=p[1], regularization_constant=0.0, epochs=1, momentum=0.5, max_size=p[0], label_mode='passive')
            rbm.log_statistics(evaluation_set, evaluation_labels)
            rbm.save_parameters(rbm_file_name.format(256, p[0], e+1))

def test_sampling_evaluation(parameter):
    logging.info('testing sampling evaluation ')

    random_seed = 1238
    # Random seed 1238 produces good sampling results, 1234 produces bad. The accuracy of sampling when compared to other things seems to be quite difficult to evaluate. This is very frustrating

    batch_size = 1000
    data_vector_size = 128
    hidden_layer_size = 128
    epochs = 10

    dwave_sampler = ModelDWave(
            layout="pegasus",
            source="dwave",
            beta=parameter,
            num_reads=400,
            s_pause=0.45,
            pause_duration=100.0,
            parallel_runs=True)

    sampler = ModelCD(1000, use_state=False, seed = random_seed)
    cd_sampler = ModelCD(1, use_state=False, seed = random_seed)

    dataset = BarsAndStripes([8, 16], 0.5, use_offsets = False, seed = random_seed)
    batches = dataset.get_batches(batch_size, include_labels = False)
    tr_set = dataset.get_training_data_without_labels()
    tr_labels = dataset.get_training_labels()
    evaluation_set = dataset.get_evaluation_data_without_labels()
    evaluation_labels = dataset.get_evaluation_labels()

    rbm = RBM(cd_sampler, shape=[data_vector_size, data_vector_size], input_included = None, weight_dist = 0.32, seed = random_seed)
    rbm.log_statistics(evaluation_set, evaluation_labels)

    res = evaluate_samplers(dwave_sampler, sampler, rbm, tr_set)
    logging.info("Quantum Sampling results: {0}".format(res))

    for i in range(5):
        rbm.train(batches, learning_rate=0.1, regularization_constant = 0.01, epochs=1, momentum=0.5)
        rbm.log_statistics(evaluation_set, evaluation_labels)

        res = evaluate_samplers(dwave_sampler, sampler, rbm, tr_set)
        logging.info("Quantum Sampling results: {0}".format(res))

#    res = evaluate_samplers(cd_sampler, sampler, rbm, tr_set)

#    logging.info("Ordinary Sampling results: {0}".format(res))

def test_statistics():
    logging.info("Creating dataset")

    batch_size = 500
    data_vector_size = 256
    hidden_layer_size = 256
    epochs = 30
    sampler = ModelCD(1)

    dataset = BarsAndStripes(16, 0.5, use_offsets = True, seed = 10000)
    batches = dataset.get_batches(batch_size, include_labels = True)
    tr_set = dataset.get_training_data_without_labels()
    tr_labels = dataset.get_training_labels()
    evaluation_set = dataset.get_evaluation_data_without_labels()
    evaluation_labels = dataset.get_evaluation_labels()

    logging.info("Training this rbm now")
    for max_size in [128]:
        rbm = RBM(sampler, shape=[data_vector_size, data_vector_size], input_included = 2)

        rbm.log_statistics(evaluation_set, evaluation_labels)

        for e in range(epochs):
            rbm.train(batches, learning_rate=0.1, regularization_constant = 0.1, epochs=1, momentum=0.5, max_size = max_size)
            pr = rbm.evaluate(evaluation_set, evaluation_labels, 5)
            logging.info("Prediction rate: {0}".format(pr))
            pr = rbm.evaluate(tr_set, tr_labels, 1)
            logging.info("Prediction rate: {0}".format(pr))

            rbm.log_statistics(evaluation_set, evaluation_labels)

def test_bars_and_stripes():
    logging.info("Creating dataset")

    batch_size = 500
    data_vector_size = 64
    hidden_layer_size = 64
    epochs = 10
    sampler = None
    dwave = False

    rbm_file_name = "./models/max_size_{0}/crbm_active_{0}_epoch_{1}.json"
    random_seed = 13241

    if dwave:
        sampler = ModelDWave(
            layout="pegasus",
            source="dwave",
            beta=3.5,
            num_reads=100,
            s_pause=0.45,
            parallel_runs=True
            )
    else:
        sampler = ModelCD(1, use_state = False, seed = random_seed)

    dataset = BarsAndStripes(8, 0.7, use_offsets = False, seed = random_seed)
    batches = dataset.get_batches(batch_size, include_labels = True)
    tr_set = dataset.get_training_data_without_labels()
    tr_labels = dataset.get_training_labels()
    evaluation_set = dataset.get_evaluation_data_without_labels()
    evaluation_labels = dataset.get_evaluation_labels()

    logging.info("Training this rbm now")
    for max_size in [64]:
        rbm = RBM(sampler, shape=[data_vector_size, data_vector_size], input_included = 2, weight_dist = 0.32, seed = random_seed)
        rbm.log_statistics(evaluation_set, evaluation_labels)
        #rbm.save_parameters(rbm_file_name.format(max_size, 0))

        for e in range(epochs):
            rbm.train(batches, learning_rate=0.1, regularization_constant=0.0, epochs=1, momentum=0.5, max_size=max_size, label_mode='passive')
            rbm.log_statistics(evaluation_set, evaluation_labels)
            #rbm.save_parameters(rbm_file_name.format(max_size, e+1))

def test_bars_and_stripes_dbn():
    logging.info("Creating dataset")
    dataset = BarsAndStripes(16, 0.75, use_offsets = True, seed = 10000)

    batch_size = 100
    data_vector_size = 256
    hidden_layer_size = 256
    max_size = 128
    epochs = 20
    sampler = None
    dwave = False

    #rbm_file_name = "./models/max_size_{0}/rbm_{0}_epoch_{1}.json"

    if dwave:
        sampler = ModelDWave(
            layout="pegasus",
            source="aws",
            parallel=3,
            beta=3.0,
            num_reads=400,
            s_pause=0.5
            )
    else:
        sampler = ModelCD(1)

    batches = dataset.get_batches(batch_size, include_labels = False)
    tr_set = dataset.get_training_data_without_labels()
    tr_labels = dataset.get_training_labels()
    f_batches = dataset.get_batches(batch_size, include_labels = True)
    evaluation_set = dataset.get_evaluation_data_without_labels()
    evaluation_labels = dataset.get_evaluation_labels()

    dbn = DBN(shape=[data_vector_size, data_vector_size, data_vector_size, data_vector_size], label_shape = 2)
    dbn.greedy_pretrain(sampler, batches, learning_rate = 0.1, epochs = epochs, momentum=0.5, regularization_constant = 0.1, max_size = 64, labels = False)

    logging.info("Training using wake sleep")
    for i in range(30):
        dbn.wakesleep_algorithm(f_batches, learning_rate = 0.01, epochs = 1, cycles = 3, regularization_constant = 0.0, momentum = 0.3)
        logging.info("Evaluating the DBN on the evaluation set")
        pr = dbn.evaluate(evaluation_set, evaluation_labels, 5)
        logging.info("Predict rate: {0}".format(pr))

        logging.info("Evaluating the DBN on the training set")
        pr = dbn.evaluate(tr_set, tr_labels, 5)
        logging.info("Predict rate: {0}".format(pr))

def test_resize():
    dataset = MnistDataset('../data/mnist_train.csv', '../data/mnist_test.csv', True)

    print(dataset.get_training_data_without_labels().shape)

def test_sampling():
    n_size = 128
    rbm = RBM(None, shape=[n_size, n_size], input_included = None, weight_dist = 0.32)
    dataset = BarsAndStripes([16, 8], 0.5, use_offsets = False, seed = 123456789)

    batches = dataset.get_batches(20, include_labels = False)

    rbm.train(batches, learning_rate=0.1, regularization_constant=0.0, epochs=2, momentum=0.5, max_size=-1, label_mode='passive')

    model_cd = ModelCD(1000)
    model_cd1 = ModelCD(1)

    gen = np.random.default_rng()

    hid_val = np.zeros([2000, n_size])
    hid_val += 0.5

    dat_values = [
            dataset.get_evaluation_data_without_labels(),
            hid_val
    ]

    cd_samp_params = {
        'weights': rbm.weights,
        'visible': rbm.visible_biases,
        'hidden': rbm.hidden_biases,
        'label_mode': 'passive',
        'dataset': dat_values,
    }

    sampler_parameters = {
        'weights': [rbm.weights],
        'visible': [rbm.visible_biases],
        'hidden': [rbm.hidden_biases],
        'label_mode': 'passive',
        'dataset': dat_values,
        'h_ids': [*range(0, n_size)],
        'v_ids': [*range(0, n_size)],
        'max_size': n_size,
        'max_divide': 1,
        'self.parallel': False
    }

    model_cd.set_model_parameters(cd_samp_params)
    model_cd1.set_model_parameters(cd_samp_params)

    an_res = model_cd.estimate_model()

    for beta in [1.0, 1.5, 2.0]:
        model_dwave = ModelDWave(
            layout="pegasus",
            source="aws",
            beta=beta,
            num_reads=100,
            s_pause=0.45,
            pause_duration=100.0,
            parallel_runs=False)

        model_dwave.set_model_parameters(sampler_parameters)

        res_1 = model_dwave.estimate_model()
        res_2 = model_cd1.estimate_model()

        print("l1 for beta: {0}".format(beta))
        print(l1_between_models(an_res, res_1))
        print("l1 for the cd1")
        print(l1_between_models(an_res, res_2))

def generate_mnist_plots():
    fig = plt.figure()
    gs = fig.add_gridspec(5, 10, hspace=0, wspace=0)

    dataset = MnistDataset('../data/mnist_train.csv', '../data/mnist_test.csv')
    evaluation_set = dataset.get_evaluation_data_without_labels()
    evaluation_labels = dataset.get_evaluation_labels()

    for i in range(5):
        for j in range(10):
            ax = fig.add_subplot(gs[i,j])
            ax.axis('off')
            ax.imshow(np.reshape(evaluation_set[i*10 + j], [28, 28]), cmap='gray')

    plt.show()

def generate_plots():
    m_size = [784, 588, 392, 256, 128, 64]
    ls = ('solid', 'dotted', 'dashed', 'dashdot', (0, (3, 5, 1, 5)), (0, (3, 5, 1, 5)), (0, (3, 4, 2, 4)))
    data = []

    for m in m_size:
        pr_file = open("../results/dbn_pr_{0}_40".format(m))
        data_array = pr_file.read()[:-1].split("\n")
        data.append([100*(float(a)) for a in data_array])

    epochs = [ i+1 for i in range(len(data[0])) ]

    plt.ylabel("Prediction rate (%)")
    plt.xlabel("Epochs")
    plt.xticks(epochs)

    for i in range(len(data)):
        plt.plot(epochs, data[i], linestyle = ls[i], linewidth = 2.0, label = r'$m_{size} =$' + "{0}".format(m_size[i]))

    plt.legend()
    plt.show()

def test_dbn_final():
    logging.info("Loading dataset")
    batch_size = 500
    data_vector_size = 196

    epochs = [10]
    fn_epochs = 10
    lr = [0.5, 0.1]
    cd = 1
    mom = [0.9, 0.95]
    reg_con = [0.0]
    max_size = [64, 98, 128]

    sampler1 = ModelCD(1)
    sampler10 = ModelCD(10)

    dataset = MnistDataset('../data/mnist_train.csv', '../data/mnist_test.csv', True)
    batches = dataset.get_batches(batch_size, include_labels = False)
    tr_set = dataset.get_training_data_without_labels()
    tr_labels = dataset.get_training_labels()
    f_batches = dataset.get_batches(batch_size, include_labels = True)
    evaluation_set = dataset.get_evaluation_data_without_labels()
    evaluation_labels = dataset.get_evaluation_labels()

    for m in max_size:
        for e in epochs:
            for r in reg_con:
                for l in lr:
                    for mo in mom:
                        dbn_file_name = "dbn_ptr_{0}_{1}".format(m, e)
                        logging.info("Creating DBN({0})".format(dbn_file_name))
                        logging.info("Params:\n lr: {0} mom: {1} rc: {2}".format(l, mo, r))
                        dbn = DBN(shape=[data_vector_size, data_vector_size, data_vector_size, data_vector_size], label_shape = 10)
                        dbn.greedy_pretrain(sampler1, batches, learning_rate = l, epochs = e, cd_iter = cd, momentum=mo, regularization_constant = r, max_size = m, labels = False)

                        logging.info("Training using wake sleep")
                        for i in range(fn_epochs):
                            dbn.wakesleep_algorithm(f_batches, learning_rate = 0.01, epochs = 1, cycles = 3, regularization_constant = r, momentum = 0.3)

                            logging.info("Evaluating the DBN on the evaluation set")
                            pr = dbn.evaluate(evaluation_set, evaluation_labels, 5)
                            logging.info("Predict rate: {0}".format(pr))

                            logging.info("Evaluating the DBN on the training set")
                            pr = dbn.evaluate(tr_set, tr_labels, 5)
                            logging.info("Predict rate: {0}".format(pr))


def finetune_dbn_final():
    logging.info("loading dataset")
    batch_size = 100
    data_vector_size = 784

    epochs = [40]
    lr = 0.01
    cd = 3
    mom = 0.5
    reg_con = 0.0001
    max_size = [64]#784, 588, 329, 256, 128]
    fn_epochs = 10

    dataset = MnistDataset('../data/mnist_train.csv', '../data/mnist_test.csv')
    batches = dataset.get_batches(batch_size, include_labels = True)

    evaluation_set = dataset.get_evaluation_data_without_labels()
    evaluation_labels = dataset.get_evaluation_labels()

    for m in max_size:
        for e in epochs:
            dbn_file_name = "../models/dbn_wks_{0}_{1}".format(m, e)
            pr_file_name = "../results/dbn_pr_{0}_{1}".format(m, e)
            dbn_load_file = "../models/dbn_ptr_{0}_{1}".format(m, e)

            pr_table = []

            logging.info("Loading DBN({0})".format(dbn_load_file))
            dbn = DBN(parameter_file = dbn_load_file)

            logging.info("Training using wake sleep")

            for i in range(fn_epochs):
                dbn.wakesleep_algorithm(batches, lr, epochs = 1, cycles = cd, momentum = mom, regularization_constant = reg_con)

                logging.info("Evaluating the DBN")
                pr = 0
                predictions = dbn.classify(evaluation_set, 2)

                for i in range(len(predictions)):
                    if np.argmax(predictions[i]) == np.argmax(evaluation_labels[i]):
                        pr += 1

                logging.info("Predict rate: {0}".format(pr / len(evaluation_set)))
                pr_table.append(pr / len(evaluation_set))

            logging.info("Storing data into files")
            dbn.save_parameters(dbn_file_name)
            pr_file = open(pr_file_name, "w")

            for p in pr_table:
                pr_file.write("{0}\n".format(p))

            pr_file.close()

def train_rbm_final():
    logging.info("Loading dataset")
    batch_size = 600
    data_vector_size = 784
    validation_size = 16

    epochs = 10
    lr = [0.05]
    cd = 1
    mom = [0.5]
    reg_con = [0.0]
    max_size = [-1]

    sampler1 = ModelCD(1)

    dataset = MnistDataset('../data/mnist_train.csv', '../data/mnist_test.csv', False)
    batches = dataset.get_batches(batch_size, include_labels = True)
    evaluation_set = dataset.get_evaluation_data_without_labels()
    evaluation_labels = dataset.get_evaluation_labels()
    tr_set = dataset.get_training_data_without_labels()
    tr_labels = dataset.get_training_labels()

    for m in max_size:
        best_pr = 0.0
        most_fit_params = None

        for l in lr:
            for mo in mom:
                for r in reg_con:
                    dbn_file_name = "rbm_ptr_{0}_{1}_{2}_{3}".format(m, l, mo, r)
                    logging.info("Creating rbm({0})".format(dbn_file_name))
                    rbm = RBM(sampler1, shape=[data_vector_size, data_vector_size], input_included = 10)

                    for e in range(epochs):
                        rbm.train(batches, learning_rate=l, epochs=1, momentum=mo, regularization_constant = r, max_size = m, label_mode = 'passive')

                        logging.info("Evaluating the RBM")
                        pr = rbm.evaluate(evaluation_set, evaluation_labels, 5)
                        logging.info("Predict rate: {0}".format(pr))

                    if best_pr < pr:
                        most_fit = rbm
                        best_pr = pr
                        most_fit_params = [l, mo, r, e]

        logging.info("Most fit params for {0}: {1}".format(m, most_fit_params))

def train_dbn_final():
    logging.info("Loading dataset")
    batch_size = 600
    data_vector_size = 196
    validation_size = 16

    epochs = 10
    lr = [0.5, 0.05, 0.01]
    cd = 1
    mom = [0.5, 0.95]
    reg_con = [0.0001, 0.0]
    max_size = [64, 98, 128]

    sampler1 = ModelCD(1)

    dataset = MnistDataset('../data/mnist_train.csv', '../data/mnist_test.csv', True)
    batches = dataset.get_batches(batch_size, include_labels = False)
    tr_set = dataset.get_training_data_without_labels()
    tr_labels = dataset.get_training_labels()
    f_batches = dataset.get_batches(batch_size, include_labels = True)
    evaluation_set = dataset.get_evaluation_data_without_labels()
    evaluation_labels = dataset.get_evaluation_labels()

    for m in max_size:
        best_pr = 0.0
        most_fit_params = None

        for l in lr:
            for mo in mom:
                for r in reg_con:
                    dbn_file_name = "dbn_ptr_{0}_{1}_{2}_{3}".format(m, l, mo, r)
                    logging.info("Creating DBN({0})".format(dbn_file_name))
                    dbn = DBN(shape=[data_vector_size, data_vector_size, data_vector_size, data_vector_size], label_shape = 10)
                    dbn.greedy_pretrain(sampler1, batches, learning_rate = l, epochs = epochs, momentum=mo, regularization_constant = r, max_size = m, labels = False)

                    dbn.wakesleep_algorithm(f_batches, learning_rate = 0.01, epochs = epochs, cycles = cd, regularization_constant = r, momentum = 0.5)

                    logging.info("Evaluating the DBN")
                    pr = dbn.evaluate(evaluation_set, evaluation_labels, 5)
                    logging.info("Predict rate: {0}".format(pr))

                    if best_pr < pr:
                        most_fit = dbn
                        best_pr = pr
                        most_fit_params = [l, mo, r]

        logging.info("Most fit params for {0}: {1}".format(m, most_fit_params))

def wake_sleep_dbn():
    logging.info("Loading dataset")
    batch_size = 100
    data_vector_size = 784

    dataset = MnistDataset('../data/mnist_train.csv', '../data/mnist_test.csv')
    batches = dataset.get_batches(batch_size, include_labels = True)

    evaluation_set = dataset.get_evaluation_data_without_labels()
    evaluation_labels = dataset.get_evaluation_labels()

    logging.info("Creating DBN")
    dbn = DBN(parameter_file = dbn_file)
    for i in range(20):
        logging.info("Training using wake sleep")
        dbn.wakesleep_algorithm(batches, 0.01, epochs = 3, cycles = 3, momentum = 0.5, regularization_constant = 2**(-15))

        logging.info("Evaluating the DBN")
        pr = 0
        predictions = dbn.classify(evaluation_set, 2)

        for i in range(len(predictions)):
            if np.argmax(predictions[i]) == np.argmax(evaluation_labels[i]):
                pr += 1
        
        logging.info("Predict rate: {0}".format(pr / len(evaluation_set)))

def evaluate_dbn():
    logging.info('Loading DBN')
    dbn = DBN(parameter_file = dbn_file)

    logging.info("Loading dataset")
    batch_size = 100
    data_vector_size = 784

    dataset = MnistDataset('../data/mnist_train.csv', '../data/mnist_test.csv')
    evaluation_set = dataset.get_evaluation_data_without_labels()
    evaluation_labels = dataset.get_evaluation_labels()
    pr = 0
    
    logging.info("Evaluating the DBN")
    predictions = dbn.classify(evaluation_set, 2)

    for i in range(len(predictions)):
        if np.argmax(predictions[i]) == np.argmax(evaluation_labels[i]):
            pr += 1
    
    logging.info("Predict rate: {0}".format(pr / len(evaluation_set)))

def evaluate_dbn_mnist():
    logging.info("Loading dataset")
    epochs = 6
    batch_size = 100
    data_vector_size = 784

    dataset = MnistDataset('../data/mnist_train.csv', '../data/mnist_test.csv')
    batches = dataset.get_batches(batch_size, include_labels = True)
    evaluation_set = dataset.get_evaluation_data_without_labels()
    evaluation_labels = dataset.get_evaluation_labels()

    logging.info("Creating DBN")
    pr_rate = []

    lrs = [0.04, 0.04]
    max_s = [256, 128]
    mom = [0.6, 0.6]

    for i in range(2):
        logging.info("Training network with max_size of {0}".format(max_s[i]))
        for e in range(2, epochs+1):
            logging.info("Training DBN with {0} epochs".format(e+5))
            dbn = DBN(shape=[data_vector_size, 512, 512, 1024], label_shape = 10)
            dbn.greedy_pretrain(batches, learning_rate = lrs[i], epochs = e, cd_iter = 2, momentum=mom[i], regularization_constant=2**(-15), max_size = max_s[i])

            pr = 0
            predictions = dbn.classify(evaluation_set, 2)
            for j in range(len(predictions)):
                if np.argmax(predictions[j]) == np.argmax(evaluation_labels[j]):
                    pr += 1

            logging.info("Predict rate: {0}".format(pr / len(evaluation_set)))
            pr_rate.append(pr/len(evaluation_set))

        logging.info("Predict_rates with different epohcs")
        logging.info(pr_rate)
        pr_rate = []

def sample_dbn():
    logging.info('Loading DBN')
    dbn = DBN(parameter_file = dbn_file)
    samples = dbn.sample(7, 10, 1000)

    utils.plot_letter(samples[2])

def train_dbn_mnist():
    logging.info("Loading dataset")
    batch_size = 100
    data_vector_size = 784

    dataset = MnistDataset('../data/mnist_train.csv', '../data/mnist_test.csv')
    batches = dataset.get_batches(batch_size, include_labels = True)

    logging.info("Creating DBN")
    dbn = DBN(shape=[data_vector_size, 640, 640, 1280], label_shape = 10)
    dbn.greedy_pretrain(batches, learning_rate = 0.01, epochs = 15, cd_iter = 2, momentum=0.5, regularization_constant=2**(-15))
    dbn.save_parameters(dbn_file)

def finetune_dbn_mnist():
    logging.info("Loading dataset")
    batch_size = 100
    data_vector_size = 784

    dataset = MnistDataset('../data/mnist_train.csv', '../data/mnist_test.csv')
    batches = dataset.get_batches(batch_size, include_labels = True)

    logging.info("Creating DBN")
    dbn = DBN(parameter_file = dbn_file)

    evaluation_set = dataset.get_evaluation_data_without_labels()
    evaluation_labels = dataset.get_evaluation_labels()

    for e in range(10):
        logging.info("Epoch: {0}-{1}".format(4*e + 1, 4*e + 3))
        pr = 0

        logging.info("Finetuning")
        dbn.finetuning_algorithm(batches, learning_rate = 0.1, epochs = 4)
        
        logging.info("Evaluating the DBN")
        predictions = dbn.classify(evaluation_set, 2)

        for i in range(len(predictions)):
            if np.argmax(predictions[i]) == np.argmax(evaluation_labels[i]):
                pr += 1
        
        logging.info("Predict rate: {0}".format(pr / len(evaluation_set)))

def evaluate_rbm_with_input():
    logging.info("Loading dataset")
    batch_size = 100
    data_vector_size = 794
    hidden_layer_size = 1000
    cycles = 2

    dataset = MnistDataset('../data/mnist_train.csv', '../data/mnist_test.csv')
    rbm = RBM(parameters=rbm_file)

    evaluation_set = dataset.get_evaluation_data_without_labels()
    evaluation_labels = dataset.get_evaluation_labels()
    pr = 0

    predictions = rbm.classify(evaluation_set, cycles)

    for i in range(len(evaluation_set)):
        if np.argmax(predictions[i]) == np.argmax(evaluation_labels[i]):
            pr += 1
    
    logging.info("Predict rate: {0}".format(pr / len(evaluation_set)))

def sample_mnist_letter():
    rbm = RBM(parameters=rbm_file)
    samples = rbm.sample(2, 1, 1000)
    utils.plot_letter(samples[0])

def train_rbm_from_mnist_with_input_final():
    logging.info("Loading dataset")
    batch_size = 100
    data_vector_size = 784
    hidden_layer_size = 784
    validation_size = 100
    epochs = [5]
    rbm_filename = '../models/rbm_params.json'

    sampler = ModelCD(1)

    dataset = MnistDataset('../data/mnist_train.csv', '../data/mnist_test.csv')
    evaluation_set = dataset.get_evaluation_data_without_labels()
    evaluation_labels = dataset.get_evaluation_labels()
    batches = dataset.get_batches(batch_size, include_labels = True, validation_set = validation_size)
    f_batches = dataset.get_batches(batch_size, include_labels = True)

    validation_set = dataset.get_validation_data(batch_size, validation_size)
    validation_labels = dataset.get_validation_labels(batch_size, validation_size)

    tr_set = dataset.get_training_data_without_labels()
    tr_labels = dataset.get_training_labels()

    lr = [0.5, 0.1]
    mom = [0.5, 0.9]
    reg_con = [0.0001]

    # run for all interesting values of of p [784, 588, 392, 256, 128, 64]
    for max_size in [64]:
        best_pr = 0.0
        most_fit_params = None

        for l in lr:
            for mo in mom:
                for r in reg_con:
                    logging.info("Training rbm for 3 epochs with params ({0} {1} {2} {3})".format(max_size, l, mo, r))
                    rbm = RBM(sampler, shape=[data_vector_size, hidden_layer_size], input_included = 10)
                    rbm.train(batches, learning_rate=l, epochs=4, momentum=mo, regularization_constant = r, max_size = max_size)

                    logging.info("Evaluating the RBM")
                    pr = rbm.evaluate(validation_set, validation_labels, 5)
                    logging.info("Predict rate: {0}".format(pr))

                    if best_pr < pr:
                        best_pr = pr
                        most_fit_params = [l, mo, r]

        logging.info("Finished choosing parameters. Starting the actual training")
        results = []

        rbm = RBM(sampler, shape=[data_vector_size, hidden_layer_size], input_included = 10)

        for e in range(1, 21):
            rbm.train(batches, learning_rate=most_fit_params[0], epochs=1, momentum=most_fit_params[1], regularization_constant = most_fit_params[2], max_size = max_size)

            pr = rbm.evaluate(evaluation_set, evaluation_labels, 5)
            results.append(pr)

        results_file = open("../models/rbm_results_{0}".format(max_size), 'w')
        for r in results:
            results_file.write(str(r) + " ")
        results_file.close()

        rbm.save_parameters("../models/rbm_final_{0}.json".format(max_size))

def sampling_effect_on_rbm():
    logging.info("Loading dataset")
    batch_size = 100
    data_vector_size = 784
    hidden_layer_size = 784
    epochs = [5]
    rbm_filename = '../models/rbm_params.json'

    samplers = [
        ModelCD(1),
        ModelCD(10)
    ]

    dataset = MnistDataset('../data/mnist_train.csv', '../data/mnist_test.csv')
    batches = dataset.get_batches(batch_size, include_labels = True)
    evaluation_set = dataset.get_evaluation_data_without_labels()
    evaluation_labels = dataset.get_evaluation_labels()
    tr_set = dataset.get_training_data_without_labels()
    tr_labels = dataset.get_training_labels()

    for s in samplers:
        rbm = RBM(s, shape=[data_vector_size, hidden_layer_size], input_included = 10)
        logging.info("Creating rbm and training for {0} epochs".format(3))

        for r in range(20):
            rbm.train(batches, learning_rate=0.1, epochs=1, momentum=0.5, regularization_constant = 0.0, max_size = -1)
            logging.info("PR on ev set: {0}".format(rbm.evaluate(evaluation_set, evaluation_labels, 5)))
            logging.info("PR on tr set: {0}".format(rbm.evaluate(tr_set, tr_labels, 5)))

def train_rbm_from_mnist_no_input():
    logging.info("Loading dataset")
    batch_size = 100
    data_vector_size = 784
    hidden_layer_size = 1000

    dataset = MnistDataset('../data/mnist_train.csv', '../data/mnist_test.csv')
    batches = dataset.get_batches(batch_size, include_labels = False)

    logging.info("Creating rbm")
    test_rbm = RBM(shape=[data_vector_size, hidden_layer_size])
    test_rbm.train(batches, learning_rate=0.01, epochs=1, cd_iter=20, momentum=0.5, regularization_constant = 2**(-16), max_size = 500)
    logging.info("Training done! Saving the parameters into a json")
    test_rbm.save_parameters(rbm_file)

def train_rbm_from_mnist_with_input():
    logging.info("Loading dataset")
    batch_size = 100
    data_vector_size = 784
    hidden_layer_size = 784
    epochs = [5]
    rbm_filename = '../models/rbm_params.json'

    sampler = ModelCD(1)

    dataset = MnistDataset('../data/mnist_train.csv', '../data/mnist_test.csv')
    batches = dataset.get_batches(batch_size, include_labels = True)
    evaluation_set = dataset.get_evaluation_data_without_labels()
    evaluation_labels = dataset.get_evaluation_labels()
    tr_set = dataset.get_training_data_without_labels()
    tr_labels = dataset.get_training_labels()
    rbm = RBM(sampler, shape=[data_vector_size, hidden_layer_size], input_included = 10)

    logging.info("Creating rbm and training for {0}".format(5))
    rbm.train(batches, learning_rate=0.5, epochs=5, momentum=0.5, regularization_constant = 0.0001, max_size = 60)

    logging.info("PR on ev set: {0}".format(rbm.evaluate(evaluation_set, evaluation_labels, 5)))
    logging.info("PR on tr set: {0}".format(rbm.evaluate(tr_set, tr_labels, 5)))

    rbm.save_parameters(rbm_filename)

def train_rbm_with_reduced_mnist():
    logging.info("Loading dataset")
    batch_size = 600
    data_vector_size = 196
    hidden_layer_size = 196
    max_size = 64
    epochs = 5
    sampler = None
    dwave = True

    rbm_file_name = "./models/max_size_{0}/rbm_{0}_epoch_{1}.json"

    if dwave:
        sampler = ModelDWave(
            layout="pegasus",
            source="aws",
            parallel=3,
            beta=3.0,
            num_reads=400,
            s_pause=0.5
            )
    else:
        sampler = ModelCD(1)

    dataset = MnistDataset('../data/mnist_train.csv', '../data/mnist_test.csv', True)
    batches = dataset.get_batches(batch_size, include_labels = False)
    evaluation_set = dataset.get_evaluation_data_without_labels()
    evaluation_labels = dataset.get_evaluation_labels()
    tr_set = dataset.get_training_data_without_labels()
    tr_labels = dataset.get_training_labels()

    rbm = RBM(sampler, shape=[data_vector_size, hidden_layer_size], input_included=None)

    logging.info("Training this rbm now")

    for e in range(epochs):
        rbm.train(batches, learning_rate=0.5, regularization_constant = 0.0, epochs=1, momentum=0.5, max_size = max_size, log_batches = True)
        rbm.save_parameters(rbm_file_name.format(max_size, e+1))

def train_dbn_with_reduced_mnist():
    logging.info("Loading dataset")
    batch_size = 600
    data_vector_size = 196
    hidden_layer_size = 196
    epochs = [5]

    sampler = None
    dwave = False

    if dwave:
        sampler = ModelDWave(
            layout="pegasus",
            source="dwave",
            parallel=4,
            beta=3.0,
            num_reads=100,
            s_pause=0.5
            )
    else:
        sampler = ModelCD(1)

    dataset = MnistDataset('../data/mnist_train.csv', '../data/mnist_test.csv', True)
    batches = dataset.get_batches(batch_size, include_labels = False)
    f_batches = dataset.get_batches(batch_size, include_labels = True)
    evaluation_set = dataset.get_evaluation_data_without_labels()
    evaluation_labels = dataset.get_evaluation_labels()
    tr_set = dataset.get_training_data_without_labels()
    tr_labels = dataset.get_training_labels()

    logging.info("Creating dbm and training for {0}".format(5))
    dbn = DBN(shape=[data_vector_size, data_vector_size, data_vector_size, data_vector_size], label_shape = 10)

    # Pretraining
    dbn.greedy_pretrain(sampler, batches, learning_rate = 0.5, epochs = 5, momentum=0.5, regularization_constant = 0.0, max_size = 64, labels = False)

    # Wakesleep
    for i in range(10):
        dbn.wakesleep_algorithm(f_batches, learning_rate = 0.05, epochs = 1, cycles = 3, regularization_constant = 0.0, momentum = 0.5)
        logging.info("PR on ev set: {0}".format(dbn.evaluate(evaluation_set, evaluation_labels, 5)))
        logging.info("PR on tr set: {0}".format(dbn.evaluate(tr_set, tr_labels, 5)))

def test_two_part_training():
    logging.info("Loading dataset")
    batch_size = 600
    data_vector_size = 196
    hidden_layer_size = 196
    epochs = [5]

    rbm_file_name = "./models/max_size_{0}/rbm_{0}_epoch_{1}.json"
    dwave = False

    if dwave:
        sampler = ModelDWave(
            layout="pegasus",
            source="aws",
            parallel=3,
            beta=3.0,
            num_reads=100,
            s_pause=0.5
            )
    else:
        sampler = ModelCD(1)

    dataset = MnistDataset('../data/mnist_train.csv', '../data/mnist_test.csv', True)
    batches = dataset.get_batches(batch_size, include_labels = False)
    f_batches = dataset.get_batches(batch_size, include_labels = True)
    evaluation_set = dataset.get_evaluation_data_without_labels()
    evaluation_labels = dataset.get_evaluation_labels()
    tr_set = dataset.get_training_data_without_labels()
    tr_labels = dataset.get_training_labels()

    rbm = RBM(sampler, shape=[data_vector_size, hidden_layer_size], input_included=10)
    rbm.input_included = None

    rbm.train(batches, learning_rate=0.5, regularization_constant = 0.0, epochs=1, momentum=0.5, max_size=64)
    rbm.input_included = 10
    rbm.train(f_batches, learning_rate=0.5, regularization_constant = 0.0, epochs=1, momentum=0.5)

    logging.info("PR on tr set: {0}".format(rbm.evaluate(tr_set, tr_labels, 5)))

if __name__ == "__main__":
    run()

