import pickle
import scipy as sp
import scipy.io
import numpy as np
import networkx as nx
import os
import re

def load_graph_from_file(fname, quiet=False):
    if 'gset' in fname:
        return load_gset_graph(fname)
    else:
        return load_graphs(os.path.dirname(fname), os.path.basename(fname), quiet)

def load_graphs(graph_dir='_graphs_eco_dqn/validation',
                graph_name='BA_20spin_m4_100graphs',
                quiet=False):
    def graph_to_nx(g):
        if isinstance(g, np.ndarray):
            g = nx.from_numpy_array(g)
        elif isinstance(g, sp.sparse.csr_matrix):
            g = nx.from_scipy_sparse_matrix(g.astype(np.float32))
        return g

    graphs_test = pickle.load(open(f'{graph_dir}/{graph_name}.pkl', 'rb'))
    graphs_test = [graph_to_nx(G) for G in graphs_test]
    try:
        sols = pickle.load(open(f'{graph_dir}/opts/cuts_{graph_name}.pkl', 'rb'))
    except FileNotFoundError:
        sols = None

    if not quiet:
        print(f'{len(graphs_test)} target graphs loaded from {graph_dir}/{graph_name}.pkl')
    return graphs_test, sols

def load_gset_graph(fname):
    """
        fname: '_gset/GXX'
    """
    if os.path.splitext(fname)[-1] == '':
        fname += ".mtx"
    graph_sp = sp.io.mmread(fname)
    graph_nx = nx.from_scipy_sparse_matrix(graph_sp.astype(np.float32))

    graph_dir, graph_name = os.path.split(fname)
    try:
        graph_id = int(re.findall('[0-9]+', graph_name)[0])
        with open(os.path.join(graph_dir, 'cuts.pkl'), 'rb') as f:
            cuts_list = pickle.load(f)
        cut = cuts_list[graph_id - 1]
    except:
        cut = None
    return [graph_nx], [cut]