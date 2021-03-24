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
    - [ ] Evaluating the results 
"""
from rbm import RBM
from dbn import DBN
from dataset import MnistDataset
from softmax import Softmax
import numpy as np
import utils
import logging
import csv

dbn_file = '../data/dbn_weight_reg_parameters.json'
rbm_file = '../data/rbm_l2_params.json'

def run():
    logging.basicConfig(level=logging.INFO)
    logging.info("Running program")

    #train_dbn_final()
    #finetune_dbn_final()
    test_dbn_final()
    #evaluate_dbn_mnist()
    #evaluate_rbm_with_input()
    #wake_sleep_dbn()
    #train_rbm_from_mnist_with_input()
    #train_rbm_from_mnist_no_input()
    #sample_mnist_letter()
    #train_dbn_mnist()
    #finetune_dbn_mnist()
    #sample_dbn()
    #test_dbn_mnist()
    #test_softmax()

def test_dbn_final():
    logging.info("Loading dataset")
    batch_size = 100
    data_vector_size = 784

    epochs = [10]
    fn_epochs = 5
    lr = [0.01]
    cd = 1
    mom = [0.5]
    reg_con = [0.1]
    max_size = [-1]

    dataset = MnistDataset('../data/mnist_train.csv', '../data/mnist_test.csv')
    batches = dataset.get_batches(batch_size, include_labels = False)
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
                        dbn = DBN(shape=[data_vector_size, 784, 784, 784], label_shape = 10)
                        dbn.greedy_pretrain(batches, learning_rate = l, epochs = e, cd_iter = cd, momentum=mo, regularization_constant = r, max_size = m, labels = False)

                        logging.info("Training using wake sleep")
                        for i in range(fn_epochs):
                            dbn.wakesleep_algorithm(f_batches, learning_rate = 0.001, epochs = 1, cycles = 1, regularization_constant = 0.01, momentum = 0.1)

                            logging.info("Evaluating the DBN")
                            pr = 0
                            predictions = dbn.classify(evaluation_set, 2)

                            for i in range(len(predictions)):
                                if np.argmax(predictions[i]) == np.argmax(evaluation_labels[i]):
                                    pr += 1
                            
                            logging.info("Predict rate: {0}".format(pr / len(evaluation_set)))

def finetune_dbn_final():
    logging.info("loading dataset")
    batch_size = 100
    data_vector_size = 784

    epochs = [12]#2, 4, 6, 8, 12]
    lr = 0.001
    cd = 3
    mom = 0.1
    reg_con = 0.0
    max_size = [256, 128]#784, 512, 392, 256, 128]
    fn_epochs = 5

    dataset = MnistDataset('../data/mnist_train.csv', '../data/mnist_test.csv')
    batches = dataset.get_batches(batch_size, include_labels = True)

    evaluation_set = dataset.get_evaluation_data_without_labels()
    evaluation_labels = dataset.get_evaluation_labels()

    for m in max_size:
        for e in epochs:
            dbn_file_name = "../models/dbn_wks_{0}e__{1}_{2}".format(fn_epochs, m, e)
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

def train_dbn_final():
    logging.info("Loading dataset")
    batch_size = 100
    data_vector_size = 784

    epochs = [2, 4, 6, 8, 12]
    lr = 0.02
    cd = 2
    mom = 0.5
    reg_con = 0.0
    max_size = [784, 512, 392, 256, 128]

    dataset = MnistDataset('../data/mnist_train.csv', '../data/mnist_test.csv')
    batches = dataset.get_batches(batch_size, include_labels = False)
    f_batches = dataset.get_batches(batch_size, include_labels = True)
    evaluation_set = dataset.get_evaluation_data_without_labels()
    evaluation_labels = dataset.get_evaluation_labels()

    for m in max_size:
        for e in epochs:
            dbn_file_name = "dbn_ptr_{0}_{1}".format(m, e)
            logging.info("Creating DBN({0})".format(dbn_file_name))
            dbn = DBN(shape=[data_vector_size, 784, 784, 784], label_shape = 10)
            dbn.greedy_pretrain(batches, learning_rate = lr, epochs = e, cd_iter = cd, momentum=mom, regularization_constant = reg_con, max_size = m, labels = False)
            dbn.save_parameters("../models/" + dbn_file_name)

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
            print(e)
            print(i)
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
    hidden_layer_size = 1024

    dataset = MnistDataset('../data/mnist_train.csv', '../data/mnist_test.csv')
    batches = dataset.get_batches(batch_size, include_labels = True)
    evaluation_set = dataset.get_evaluation_data_without_labels()
    evaluation_labels = dataset.get_evaluation_labels()

    logging.info("Creating rbm")
    rbm = RBM(shape=[data_vector_size, hidden_layer_size], input_included = 10)

    for e in range(10):
        rbm.train(batches, learning_rate=0.02, epochs=2, cd_iter=2, momentum=0.6, regularization_constant = 2**(-15), max_size = 512)
        pr = 0

        predictions = rbm.classify(evaluation_set, 2)

        for i in range(len(evaluation_set)):
            if np.argmax(predictions[i]) == np.argmax(evaluation_labels[i]):
                pr += 1
        
        logging.info("Predict rate: {0}".format(pr / len(evaluation_set)))

if __name__ == "__main__":
    run()
