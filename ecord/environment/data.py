from collections import namedtuple

import networkx as nx
import numpy as np
import torch
from scipy.sparse.linalg import eigs, eigsh
from torch_geometric.data import Batch as Batch_tg
from torch_geometric.data import Data
from torch_geometric.transforms import Compose
from torch_geometric.utils import degree
from torch_geometric.utils import get_laplacian, to_scipy_sparse_matrix
from torch_scatter import scatter_min, scatter_max, scatter_mean, scatter_std, scatter_sum

Batch = namedtuple('Batch', ['nx', 'tg', 'info'])

BatchInfo = namedtuple('BatchInfo',
                       field_names=['num_nodes', 'batch_offset', 'batch_to_flat_idxs'])

class GraphBatcher():

    def __init__(self,
                 add_normalised_degree=False,
                 add_degree_profile=False,
                 add_weight_profile=False,
                 norm_features=False,
                 add_laplacian_PEs=False,
                 ):
        add_node_profiles = LocalNodeProfile(
            add_normalised_degree,
            add_degree_profile,
            add_weight_profile,
            norm_features,
        )
        if any([add_normalised_degree, add_degree_profile, add_weight_profile, norm_features]):
            cat = True
        else:
            cat = False
        if add_laplacian_PEs:
            lap_PEs = LaplacianPEs(
                k=3,
                randomise_signs=True,
                normalization=None,
                is_undirected=False,
                cat = cat,
            )
            self.add_node_data = Compose([
                add_node_profiles,
                lap_PEs,
            ])
        else:
            self.add_node_data = add_node_profiles

    def batch_tg_from_data_list(self, data_list):
        data_list = [self.add_node_data(data) for data in data_list]
        batch_tg = Batch_tg.from_data_list(data_list)

        node_degree = degree(batch_tg.edge_index[0], num_nodes=batch_tg.num_nodes).long()
        batch_tg.edges_by_node = batch_tg.edge_index.split(node_degree.tolist(), 1)
        batch_tg.edge_attrs_by_node = batch_tg.edge_attr.squeeze().split(node_degree.tolist(), 0)

        batch_tg.edges_and_attrs_by_node = torch.cat([batch_tg.edge_index, batch_tg.edge_attr.T], 0).split(
            node_degree.tolist(), 1)

        batch_tg.degree = node_degree
        degree_max = torch.zeros(batch_tg.num_nodes)
        degree_max[scatter_max(node_degree, batch_tg.batch)[1]] = 1
        batch_tg.degree_max = degree_max

        return batch_tg

    def __call__(self, batch_nx, return_tg_only=False) -> Batch:
        data_list = [
            GraphData.from_networkx(G, node_dim=None, glob_dim=None)
            for G in batch_nx
        ]

        batch_tg = self.batch_tg_from_data_list(data_list)

        if return_tg_only:
            return Batch(tg=batch_tg, nx=None, info=None)
        else:
            batch_info = get_batch_info(batch_tg.batch, [G.number_of_nodes() for G in batch_nx])
            return Batch(batch_nx, batch_tg, batch_info)

def get_batch_info(batch, num_nodes_list):
    if not torch.is_tensor(num_nodes_list):
        num_nodes_list = torch.tensor(num_nodes_list)

    batch_offset = torch.roll(torch.cumsum(num_nodes_list, 0), 1)
    batch_offset[0] = 0

    batch_to_flat_idxs = (batch, torch.cat([torch.arange(n) for n in num_nodes_list]))

    return BatchInfo(num_nodes_list, batch_offset, batch_to_flat_idxs)

class GraphData(Data):
    '''
    Extension of torch_geometric.data.Data, where for a single graph we want the following data:
        x : [num_nodes, num_features]
        u : [num_features]
        edge_index : [2, num_edges]
        edge_attr : [1, num_edges]
    '''

    def __init__(self, x=None, u=None, edge_index=None, edge_attr=None, num_nodes=None):
        super().__init__()

        self.x = x
        self.u = u
        self.edge_index = edge_index
        self.edge_attr = edge_attr
        self.num_nodes = num_nodes

    @staticmethod
    def from_networkx(G, node_dim, glob_dim=None):
        G = nx.convert_node_labels_to_integers(G)
        G = G.to_directed() if not nx.is_directed(G) else G

        x = torch.ones(G.number_of_nodes(), 1)
        if glob_dim is not None:
            u = torch.zeros(glob_dim)
        else:
            u = None
        edge_index = torch.LongTensor(list(G.edges)).t().contiguous().view(2, -1)
        edge_weights = nx.get_edge_attributes(G, "weight")
        edge_attr = torch.tensor(list(edge_weights.values())).unsqueeze(1)

        return GraphData(x, u, edge_index, edge_attr, G.number_of_nodes())

def quick_batch_tg_copy(batch_tg):
    '''
    A quick and dirty batch_tg.clone() where batch_tg is a pytorch_geometric Batch.
    Why? We return the batch_tg as part of the environmental observation, as it is
    used as input to our network.  However, if we pass by reference, then once the
    environment takes a step, the passed observation will also change...not good.

    This is resolved by passing a copy (batch_tg.clone()), however, *for reasons
    that completely elude me*, if we do this our entire training loop gets an order
    of magnitude slower (partly due to expensive .clone() calls, however there appears
    to be some knock on slow-down effects that I can't identify).  We can't simply create
    a new Batch() with the required data (e.g. node observation tensor, edge indices etc)
    because Batch remembers *at initialisation* the individual graphs it is made up from.
    Ultimately, if we want to be able to reconstruct these individual inputs (which we do
    to, for example, add them to a replay buffer), then we need to also pass along all of
    this remembered information.

    Therefore, the below code creates a new Batch and populates it with the required
    information.  However it is not a full clone, which means it is (1) about an order
    of magnitude faster that .clone() and (2) doesn't result in the *completely
    unexplained* slow-down.
    '''
    # batch_new = Batch(batch_tg.batch) # torch_geometric.__version__ <= 1.6.3
    batch_new = Batch_tg(batch_tg.batch, batch_tg.ptr) # For torch_geometric.__version__ > 1.6.3
    batch_new.__data_class__ = batch_tg.__data_class__
    batch_new.__slices__ = batch_tg.__slices__
    batch_new.__cumsum__ = batch_tg.__cumsum__
    batch_new.__cat_dims__ = batch_tg.__cat_dims__
    batch_new.__num_nodes_list__ = batch_tg.__num_nodes_list__
    batch_new.__num_graphs__ = batch_tg.__num_graphs__
    batch_new.x = batch_tg.x
    batch_new.u = batch_tg.u
    batch_new.edge_index = batch_tg.edge_index
    batch_new.edge_attr = batch_tg.edge_attr
    batch_new.edges_by_node = batch_tg.edges_by_node
    batch_new.edge_attrs_by_node = batch_tg.edge_attrs_by_node
    batch_new.edges_and_attrs_by_node = batch_tg.edges_and_attrs_by_node
    batch_new.degree = batch_tg.degree
    batch_new.degree_max = batch_tg.degree_max
    batch_new.batch = batch_tg.batch
    return batch_new

class LocalNodeProfile(object):

    def __init__(self,
                 add_normalised_degree=False,
                 add_degree_profile=False,
                 add_weight_profile=False,
                 norm_features=False,
                 ):
        self.add_normalised_degree = add_normalised_degree
        self.add_degree_profile = add_degree_profile
        self.add_weight_profile = add_weight_profile
        self.norm_features = norm_features

    def __get_profile(self, node_feature, row, col, N):
        node_feature = node_feature.float()

        if self.norm_features:
            node_feature /= node_feature.max()

        node_feature_col = node_feature[col]

        min_nf, _ = scatter_min(node_feature_col, row, dim_size=N)
        max_nf, _ = scatter_max(node_feature_col, row, dim_size=N)
        mean_nf = scatter_mean(node_feature_col, row, dim_size=N)
        std_nf = scatter_std(node_feature_col, row, dim_size=N)

        return torch.stack([node_feature, min_nf, max_nf, mean_nf, std_nf], dim=1)

    def __call__(self, data):
        row, col = data.edge_index
        N = data.num_nodes

        features = []

        if self.add_normalised_degree or self.add_degree_profile:
            deg = degree(row, N, dtype=torch.float)

        if self.add_normalised_degree:
            norm_deg = (deg - deg.mean()) / deg.std()
            features.append(norm_deg.unsqueeze(-1))

        if self.add_degree_profile:
            features.append(
                self.__get_profile(deg, row, col, N)
            )

        if self.add_weight_profile:
            weight = scatter_sum(data.edge_attr.squeeze(), row, dim_size=N)
            features.append(
                self.__get_profile(weight, row, col, N)
            )

        if len(features)>=2:
            x = torch.cat(features, dim=-1)
        elif len(features)==1:
            x = features[0]
        else:
            x = torch.ones(N, 1)

        x = x.float()

        data.x = x

        return data

    def __repr__(self):
        return '{}()'.format(self.__class__.__name__)


class LaplacianPEs(object):
    r"""Computes the highest eigenvalue of the graph Laplacian given by
    :meth:`torch_geometric.utils.get_laplacian`.

    Args:
        normalization (str, optional): The normalization scheme for the graph
            Laplacian (default: :obj:`None`):

            1. :obj:`None`: No normalization
            :math:`\mathbf{L} = \mathbf{D} - \mathbf{A}`

            2. :obj:`"sym"`: Symmetric normalization
            :math:`\mathbf{L} = \mathbf{I} - \mathbf{D}^{-1/2} \mathbf{A}
            \mathbf{D}^{-1/2}`

            3. :obj:`"rw"`: Random-walk normalization
            :math:`\mathbf{L} = \mathbf{I} - \mathbf{D}^{-1} \mathbf{A}`
        is_undirected (bool, optional): If set to :obj:`True`, this transform
            expects undirected graphs as input, and can hence speed up the
            computation of the largest eigenvalue. (default: :obj:`False`)
    """

    def __init__(self, k, randomise_signs=True, normalization=None, is_undirected=False, cat=True):
        assert normalization in [None, 'sym', 'rw'], 'Invalid normalization'
        self.k = k
        self.randomise_signs = randomise_signs
        self.normalization = normalization
        self.is_undirected = is_undirected
        self.cat = cat

    def __call__(self, data):
        edge_weight = data.edge_attr
        if edge_weight is not None and edge_weight.numel() != data.num_edges:
            edge_weight = None

        edge_index, edge_weight = get_laplacian(data.edge_index, edge_weight,
                                                self.normalization,
                                                num_nodes=data.num_nodes)

        L = to_scipy_sparse_matrix(edge_index, edge_weight, data.num_nodes)

        eig_fn = eigs
        if self.is_undirected and self.normalization != 'rw':
            eig_fn = eigsh

        evecs = eig_fn(L.asfptype(), k=self.k, which='LM', return_eigenvectors=True)[1]
        evecs = torch.from_numpy(np.real(evecs))

        if self.randomise_signs:
            signs = 2 * torch.randint(0, 2, evecs.shape[:-1]).unsqueeze(-1) - 1
            evecs *= signs

        if data.x is not None and self.cat:
            data.x = torch.cat([data.x, evecs], dim=-1)
        else:
            data.x = evecs

        return data