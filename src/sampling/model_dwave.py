"""
Class for estimating the model distribution of an RBM with dwave quantum annealing
"""

import json

import numpy as np
from networkx import Graph

from dwave.embedding.chimera import find_biclique_embedding
from dwave.embedding import embed_ising
from dwave_networkx import chimera_graph
from dimod.reference.samplers import SimulatedAnnealingSampler
from dwave.inspector import show
from dwave.system import DWaveSampler, FixedEmbeddingComposite

from braket.ocean_plugin import BraketDWaveSampler

from sampling.model import Model

from sampling.utils import get_destination_folder

class ModelDWave(Model):
    """
    Base class for model for DWave machines with various layouts
    """
    def __init__(self, layout, source = "dwave", parallel = 1, beta=1.0, num_reads = 100, s_pause = 0.5):
        super(ModelDWave, self).__init__("model_dwave")

        if layout not in ['chimera', 'pegasus']:
            raise Exception('Layout not a valid one (0)'.format(layout))

        self.layout = layout
        self.source = source
        self.sampler = self.create_sampler()
        self.parallel = parallel
        self.beta = beta
        self.num_reads = num_reads
        self.s_pause = s_pause

    def estimate_model(self):
        """
        Estimate the model distribution by sampling quantum annealing device
        """
        embedding = self.generate_embedding()

        emb_problem = FixedEmbeddingComposite(self.sampler, embedding)

        h_bias, j_coupling = self.generate_couplings()

        a_schedule = [
                [0.0, 0.0],
                [50.0, self.s_pause],
                [1050.0, self.s_pause],
                [1100.0, 1.0]
                ]

        response = emb_problem.sample_ising(
            h_bias,
            j_coupling,
            return_embedding = True,
            num_reads = self.num_reads,
            num_spin_reversal_transforms = 5,
            anneal_schedule = a_schedule)

        vis_states, hid_states = self.extract_values_from_samples(response)

        return  [vis_states, hid_states]

    def create_sampler(self):
        """
        Initialize the sampler for the sampling process
        """
        if self.source == "dwave":
            if self.layout == "chimera":
                return DWaveSampler(solver={'topology__type': 'chimera'})
            elif self.layout == "pegasus":
                return DWaveSampler(solver={'name': 'Advantage_system4.1'})
        elif self.source == "aws":
            return BraketDWaveSampler(get_destination_folder())
        else:
            raise Exception("Source not defined correctly!")

    def extract_values_from_samples(self, response):
        """
        Take a dwave sampleset and extract hidden and visible states from them
        """
        n_samples = len(response.samples())
        vis_states = np.zeros([n_samples * self.parallel, len(self.visible)])
        hid_states = np.zeros([n_samples * self.parallel, len(self.hidden)])

        for pr in range(self.parallel):
            for i in range(len(response.samples())):
                for vis_id in range(len(self.visible)):
                    vis_states[n_samples * pr + i, vis_id] = response.samples()[i]['v_{0}_{1}'.format(pr, vis_id)] / 2 + 0.5

                for hid_id in range(len(self.hidden)):
                    hid_states[n_samples * pr + i, hid_id] = response.samples()[i]['h_{0}_{1}'.format(pr, hid_id)] / 2 + 0.5

        return vis_states, hid_states

    def generate_embedding(self):
        """
        Generate dwave couplings for the RBM in question
        """
        if self.layout == 'chimera':
            return self.generate_chimera_embedding()
        elif self.layout == 'pegasus':
            return self.generate_pegasus_embedding()
        else:
            raise Exception('No sampler set!')

    def generate_pegasus_embedding(self):
        """
        Generate pegasus embedding from the mined embedding file
        """
        embedding = None
        source_file = None

        if self.source == 'dwave':
            if self.parallel == 4:
                source_file = 'sampling/models/advantage_parallel_mapping_4.json'
            else:
                source_file = 'sampling/models/advantage_64_mapping.json'
        elif self.source == 'aws':
            source_file = 'sampling/models/aws_128_mapping.json'

        with open(source_file, 'r') as advantage_file:
            embedding = json.load(advantage_file)['mapping']

        return embedding

    def generate_chimera_embedding(self):
        """
        Generate embedding for chimera layout
        """
        v_labels = []
        h_labels = []

        for i in range(len(self.visible)):
            v_labels.append("v_0_{0}".format(i))

        for i in range(len(self.hidden)):
            h_labels.append("h_0_{0}".format(i))

        t_graph = self.generate_graph()
        vis, hid = find_biclique_embedding(v_labels, h_labels, 16, target_edges=t_graph.edges)

        return vis | hid

    def generate_couplings(self):
        """
        Generate coupling parameters h and J
        """
        j_couplings = {}
        h_mag = {}

        for pr in range(self.parallel):
            for i in range(len(self.visible)):
                for j in range(len(self.hidden)):
                    j_couplings[('v_{0}_{1}'.format(pr, i), 'h_{0}_{1}'.format(pr, j))] = -self.weights[i, j] / 4

            for i in range(len(self.visible)):
                h_mag['v_{0}_{1}'.format(pr, i)] = -self.visible[i] / 2

                for j in range(len(self.hidden)):
                    h_mag['v_{0}_{1}'.format(pr, i)] -= self.weights[i, j] / 4

            for j in range(len(self.hidden)):
                h_mag['h_{0}_{1}'.format(pr, j)] = -self.hidden[j] / 2

                for i in range(len(self.hidden)):
                    h_mag['h_{0}_{1}'.format(pr, j)] -= self.weights[i, j] / 4

        for h_key in h_mag.keys():
            h_mag[h_key] /= self.beta

        for j_key in j_couplings.keys():
            j_couplings[j_key] /= self.beta

        return h_mag, j_couplings

    def generate_graph(self):
        """
        Generate the chimera graph for DWave 2000Q
        """
        with open('sampling/models/dwave2000q_6.json') as json_file:
            graph_json = json.load(json_file)

        dwave_6 = Graph()

        dwave_6.add_nodes_from(graph_json['qubits'])
        dwave_6.add_edges_from(graph_json['couplings'])

        return dwave_6

    def get_samples_num(self):
        """
        Get the amount of samples estimate_model returns
        """
        return self.parallel * self.num_reads


