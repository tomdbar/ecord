import networkx as nx
import scipy as sp
import torch


def _get_laplacian_PEs(G, k=1):
    L = sp.sparse.eye(G.number_of_nodes()) - nx.linalg.normalized_laplacian_matrix(G)
    evecs = sp.sparse.linalg.eigsh(L.asfptype(), k=k, which='LM')[1]
    return torch.from_numpy(evecs)

def get_laplacian_PEs(batch_nx, k=1, num_tradj=-1, randomise_signs=True):
    lap_PEs = torch.stack([_get_laplacian_PEs(G.to_undirected(), k) for G in batch_nx])
    if num_tradj >0:
        # lap_PEs : [batch, num_nodes, k]
        lap_PEs = lap_PEs[:,:,None,:].repeat(1,1,num_tradj,1)
    if randomise_signs:
        signs = 2*torch.randint(0,2,lap_PEs.shape[:-1]).unsqueeze(-1) - 1
        lap_PEs *= signs
    return lap_PEs