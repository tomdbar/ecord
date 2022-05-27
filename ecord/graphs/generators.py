import networkx as nx
from abc import ABC, abstractmethod
from collections.abc import Sequence
from enum import Enum
import numpy as np


class EdgeType(Enum):
    CONSTANT = 0  # All 1.
    CONSTANT_NEG = 1 # All -1
    SIGNED = 2  # Randomly +/- 1.
    RANDOM = 3  # Uniform distribution between [-1,1].

class _GraphGeneratorBase(ABC):

    def get_fixed_generator(self, n):
        return FixedGraphGenerator([G for G in self.generate(n)])

    def generate(self, n):
        for _ in range(n):
            yield self.get()

    @abstractmethod
    def get(self):
        raise NotImplementedError


class FixedGraphGenerator(_GraphGeneratorBase):

    def __init__(self, graphs):
        if not isinstance(graphs, Sequence):
            graphs = [graphs]
        self.graphs = graphs
        self.idx, self.size = 0, len(graphs)

    def get(self):
        G = self.graphs[self.idx]
        self.idx = (self.idx + 1) % self.size
        return G

class _NXGraphGenerator(_GraphGeneratorBase):

    def __init__(self,
                 nx_generator_callable,
                 nx_generator_args_callable,
                 edge_type=EdgeType.CONSTANT
                 ):

        self.nx_generator_callable = nx_generator_callable
        self.nx_generator_args = nx_generator_args_callable
        self.edge_type = edge_type

    def set_edge_weights(self, G):
        edges = G.edges()

        if self.edge_type is EdgeType.CONSTANT:
            attr = 1
        elif self.edge_type is EdgeType.CONSTANT_NEG:
            attr = -1
        elif self.edge_type is EdgeType.SIGNED:
            weights = 2 * np.random.randint(0, 2, len(edges)) - 1
            attr = dict([(e, w) for e, w in zip(edges, weights)])
        elif self.edge_type is EdgeType.RANDOM:
            weights = np.random.uniform(-1, 1, size=(len(edges))).astype(dtype=np.float32)
            attr = dict([(e, w) for e, w in zip(edges, weights)])
        else:
            raise ValueError(f"{self.edge_type} is not a recognised value of EdgeType.")

        nx.set_edge_attributes(G, attr, "weight")

        return G

    def get(self):
        G = self.nx_generator_callable(self.nx_generator_args())
        return self.set_edge_weights(G)


class ErdosRenyiGraphGenerator(_NXGraphGenerator):

    def __init__(self,
                 num_nodes=[15, 25],
                 p_connection=[0.1, 0],
                 edge_type=EdgeType.CONSTANT):

        if not isinstance(num_nodes, Sequence):
            num_nodes = [num_nodes, num_nodes]
        if not isinstance(p_connection, Sequence):
            p_connection = [p_connection, 0]
        assert len(num_nodes) == 2, "num_nodes must have length 2 denoting a range."
        assert len(p_connection) == 2, "p_connection must have length 2 denoting [mean, std]."

        self.num_nodes = num_nodes
        self.p_connection = p_connection

        super().__init__(lambda args: nx.erdos_renyi_graph(args[0], args[1]),
                         lambda: [np.random.randint(self.num_nodes[0], self.num_nodes[1] + 1),
                                  np.clip(np.random.normal(self.p_connection[0], self.p_connection[1]), 0, 1)],
                         edge_type)

class BarabasiAlbertGraphGenerator(_NXGraphGenerator):

    def __init__(self,
                 num_nodes=[15, 25],
                 m_insertion_edges=[4, 0],
                 edge_type=EdgeType.SIGNED):

        if not isinstance(num_nodes, Sequence):
            num_nodes = [num_nodes, num_nodes]
        if not isinstance(m_insertion_edges, Sequence):
            m_insertion_edges = [m_insertion_edges, 0]
        m_insertion_edges = [int(x) for x in m_insertion_edges]
        assert len(num_nodes) == 2, "num_nodes must have length 2 denoting a range."
        assert len(m_insertion_edges) == 2, "m_insertion_edges must have length 2 denoting [mean, range]."

        self.num_nodes = num_nodes
        self.m_insertion_edges = m_insertion_edges

        def get_m_insertion_edges():
            if self.m_insertion_edges[1] == 0:
                return self.m_insertion_edges[0]
            else:
                return np.random.randint(self.m_insertion_edges[0]-self.m_insertion_edges[1],
                                         self.m_insertion_edges[0]+self.m_insertion_edges[1])

        super().__init__(lambda args: nx.barabasi_albert_graph(args[0], args[1]),
                         lambda: [np.random.randint(self.num_nodes[0], self.num_nodes[1] + 1),
                                  get_m_insertion_edges()],
                         edge_type)