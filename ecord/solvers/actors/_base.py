import itertools
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass
from typing import Optional

import torch
from torch.distributions import Bernoulli


def _clone(x):
    if x is None:
        return x
    else:
        return x.clone()


def _clone_to(x, device):
    if x is None:
        return x
    else:
        return x.clone().to(device)

@dataclass
class ActorState:
    hidden_embeddings: torch.Tensor
    init_node_embeddings: torch.Tensor

    last_selected_node_embeddings: Optional[torch.Tensor] = None
    cell_embeddings: Optional[torch.Tensor] = None
    graph_embeddings: Optional[torch.Tensor] = None

    def to(self, *args, **kwargs):
        self.hidden_embeddings = self.hidden_embeddings.to(*args, **kwargs)
        self.init_node_embeddings = self.init_node_embeddings.to(*args, **kwargs)
        if self.last_selected_node_embeddings is not None:
            self.last_selected_node_embeddings = self.last_selected_node_embeddings.to(*args, **kwargs)
        if self.cell_embeddings is not None:
            self.cell_embeddings.to(*args, **kwargs)
        if self.graph_embeddings is not None:
            self.graph_embeddings.to(*args, **kwargs)
        return self

    def cpu(self):
        self.to('cpu')
        return self

    def cuda(self):
        self.to('cuda')
        return self

    def detach(self):
        self.hidden_embeddings = self.hidden_embeddings.detach()
        self.init_node_embeddings = self.init_node_embeddings.detach()
        if self.last_selected_node_embeddings is not None:
            self.last_selected_node_embeddings = self.last_selected_node_embeddings.detach()
        if self.cell_embeddings is not None:
            self.cell_embeddings.detach()
        if self.graph_embeddings is not None:
            self.graph_embeddings.detach()
        return self

    def detach_(self):
        self.hidden_embeddings = self.hidden_embeddings.detach_()
        self.init_node_embeddings = self.init_node_embeddings.detach_()
        if self.last_selected_node_embeddings is not None:
            self.last_selected_node_embeddings = self.last_selected_node_embeddings.detach_()
        if self.cell_embeddings is not None:
            self.cell_embeddings.detach_()
        if self.graph_embeddings is not None:
            self.graph_embeddings.detach_()

    def clone(self):
        return ActorState(
            hidden_embeddings=_clone(self.hidden_embeddings),
            init_node_embeddings=_clone(self.init_node_embeddings),
            last_selected_node_embeddings=_clone(self.last_selected_node_embeddings),
            cell_embeddings=_clone(self.cell_embeddings),
            graph_embeddings=_clone(self.graph_embeddings),
        )


class _ActorBase(ABC):
    tabu_masked_val = -1e3

    def __init__(self,
                 gnn,
                 rnn,
                 gnn2rnn=None,
                 node_encoder=None,
                 node_classifier=None,
                 add_glob_to_nodes=False,
                 add_glob_to_obs=False,
                 detach_gnn_from_rnn=False,
                 device=None,
                 ):

        self.gnn = gnn
        self.rnn = rnn
        self.gnn2rnn = gnn2rnn
        self.node_encoder = node_encoder
        self.node_classifier = node_classifier
        self.detach_gnn_from_rnn = detach_gnn_from_rnn
        self.add_glob_to_nodes = add_glob_to_nodes
        self.add_glob_to_obs = add_glob_to_obs

        if device is not None:
            self.set_device(device)

        self.reset_state()

    def parameters(self):
        params_list = [self.gnn.parameters(), self.rnn.parameters()]
        if self.gnn2rnn is not None:
            params_list.append(self.gnn2rnn.parameters())
        if self.node_encoder is not None:
            params_list.append(self.node_encoder.parameters())
        if self.node_classifier is not None:
            params_list.append(self.node_classifier.parameters())
        return itertools.chain(*params_list)

    def reset_state(self):
        self.init_node_embeddings = None
        self.node_embeddings = None
        self._last_selected_node_embeddings = None
        self.rnn.hidden_embeddings = None
        if hasattr(self.rnn, 'cell_embeddings'):
            self.rnn.cell_embeddings = None
        if hasattr(self.rnn, 'graph_embeddings'):
            self.rnn.graph_embeddings = None

    def get_state(self):
        state = self.rnn.get_internal_state()
        state['init_node_embeddings'] = _clone(self.init_node_embeddings)
        state['last_selected_node_embeddings'] = _clone(self._last_selected_node_embeddings)
        return ActorState(**state)

    def set_state(self, actor_state, use_target_network=False, overwite_init_node_embeddings=False, reset_internal_state=False):
        rnn_embeddings = {}
        for lab, e in zip(['hidden_embeddings', 'cell_embeddings', 'graph_embeddings'],
                          [actor_state.hidden_embeddings, actor_state.cell_embeddings, actor_state.graph_embeddings]):
            if e is not None:
                if lab == 'graph_embeddings' or not reset_internal_state:
                    rnn_embeddings[lab] = _clone_to(e, self.device)
                else:
                    rnn_embeddings[lab] = torch.zeros_like(e, device=self.device)

        if use_target_network:
            rnn = self.rnn_target if hasattr(self, "rnn_target") else self.rnn
        else:
            rnn = self.rnn

        rnn.set_internal_state(**rnn_embeddings)

        self._last_selected_node_embeddings = _clone_to(actor_state.last_selected_node_embeddings, self.device)

        if overwite_init_node_embeddings:
            self.init_node_embeddings = _clone(actor_state.init_node_embeddings).to(self.device)

    def initialise_node_embeddings(self, batch, num_tradj, use_target_network=False):
        if use_target_network:
            gnn = self.gnn_target if hasattr(self, "gnn_target") else self.gnn
        else:
            gnn = self.gnn

        node_embeddings = gnn(
            x = batch.tg.x,
            edge_index = batch.tg.edge_index,
            edge_attr = batch.tg.edge_attr,
            u = batch.tg.u,
            batch = batch.tg.batch,
            add_self_loops=False,
            out_device=self.device
        )

        self.gnn_node_embeddings = node_embeddings.unsqueeze(1).repeat(1, num_tradj, 1)
        if self.detach_gnn_from_rnn:
            node_embeddings = node_embeddings.detach()

        if self.gnn2rnn is not None:
            node_embeddings = self.gnn2rnn(node_embeddings)

        self.init_node_embeddings = node_embeddings.unsqueeze(1).repeat(1, num_tradj, 1)

    def initialise_embeddings(self, batch, num_tradj, use_target_network=False):
        '''Initialise per node and per graph embeddings.

            node_embeddings : [num_nodes_in_graphs, num_tradj, dim_node_embedding]
            rnn.graph_embeddings : [num_graphs, num_tradj, dim_graph_embedding]
            _last_selected_node_embeddings : [num_graphs, num_tradj, dim_node_embedding]
        '''
        if use_target_network:
            gnn = self.gnn_target if hasattr(self, "gnn_target") else self.gnn
            rnn = self.rnn_target if hasattr(self, "rnn_target") else self.rnn
        else:
            gnn = self.gnn
            rnn = self.rnn

        # node_embeddings --> [num_graphs, dim_graph_embedding]
        # Note, only one tradj so we can efficiently set graph embeddings in next line,
        # after which we will copy the node_embeddings across all tradjectories in the
        # line after that.
        node_embeddings = gnn(
            x=batch.tg.x,
            edge_index=batch.tg.edge_index,
            edge_attr=batch.tg.edge_attr,
            u=batch.tg.u,
            batch=batch.tg.batch,
            add_self_loops=False,
            out_device=self.device
        )

        self.gnn_node_embeddings = node_embeddings.unsqueeze(1).repeat(1, num_tradj, 1)
        if self.detach_gnn_from_rnn:
            node_embeddings = node_embeddings.detach()

        if self.gnn2rnn is not None:
            node_embeddings = self.gnn2rnn(node_embeddings)

        # rnn.graph_embeddings --> [num_graphs, num_tradj, dim_graph_embedding]
        rnn.init_embeddings(node_embeddings, batch, num_tradj)

        # node_embeddings --> [num_nodes_in_graphs, num_tradj, dim]
        self.init_node_embeddings = node_embeddings.unsqueeze(1).repeat(1, num_tradj, 1)

        # _last_selected_node_embeddings --> [num_graphs, num_tradj, dim_node_embedding + dim_node_features]
        if type(rnn.hidden_embeddings) is list:
            sz = rnn.hidden_embeddings[-1].size(0)
        else:
            sz = rnn.hidden_embeddings.size(0)
        self._last_selected_node_embeddings = torch.zeros(
            sz,
            num_tradj,
            self.rnn.dim_node_embedding
        )

        self.log = defaultdict(list)

    def get_node_dist_from_gnn_embeddings(self):
        if not self.node_classifier:
            return None
        else:
            # Probs: [num_nodes_in_graphs, num_tradj, num_prob_maps]
            logits = self.node_classifier(self.gnn_node_embeddings)
            return Bernoulli(logits=logits)

    def _update_internal_state(self, obs, use_target_network=False):
        '''Updates the internal state of the RNN, based on the last selected
        node embeddings (at t) and an observation of the environment (at t+1).
        '''
        # Update combining last chosen actions and oberved rewards.
        if self._last_selected_node_embeddings is not None:
            # Only update the internal state if the last selected node embeddings are there.
            # Note that after updating, we delete the cached embeddings to ensure we only use them once.
            if obs.glob_features is not None and self.add_glob_to_obs:
                # obs_embeddings --> [num_graphs, num_tradj, dim+dim_glob_features]
                obs_embeddings = torch.cat(
                    [self._last_selected_node_embeddings.to(self.device), obs.glob_features.to(self.device)],
                    dim=-1
                )
            else:
                obs_embeddings = self._last_selected_node_embeddings.to(self.device)

            self._last_selected_node_embeddings = None

            if use_target_network:
                self.rnn_target.update_embeddings(obs_embeddings.to(self.device))
            else:
                self.rnn.update_embeddings(obs_embeddings.to(self.device))

    def _get_node_features(self, obs):
        node_features = obs.node_features.to(self.device)
        if obs.glob_features is not None and self.add_glob_to_nodes:
            node_features = torch.cat([node_features, obs.glob_features.to(self.device)[obs.batch_idx]], dim=-1)
        if self.node_encoder is not None:
            node_features = self.node_encoder(node_features)
        return node_features

    def _get_node_embeddings(self, obs, *args, **kwargs):
        '''Get per-node embeddings.

        Concatenates the node-embeddings of the GNN (self.node_embeddings) and
        the per-node features from the observation (obs.node_features).
        '''
        if obs.node_features is not None:
            # node_embeddings --> [num_nodes, num_tradj, dim+dim_node_features]
            node_embeddings = torch.cat(
                [self.init_node_embeddings.to(self.device), self._get_node_features(obs, *args, **kwargs)],
                dim=-1
            )
        else:
            node_embeddings = self.init_node_embeddings.to(self.device)

        return node_embeddings

    def _cache_action_embeddings(self, embeddings, actions, stop_grad=False):
        '''Caches the embeddings corresponding to the selected actions
        internally for use at the next time-step.

        embeddings : [num_nodes, num_tradj, dim_node_embeddings+dim_node_features]
        actions : [num_graphs, num_tradj]
        '''
        if stop_grad:
            embeddings, actions = embeddings.detach(), actions.detach()
        self._last_selected_node_embeddings = embeddings.gather(
            dim=0,
            index=actions.unsqueeze(-1).repeat(1, 1, embeddings.size(-1))
        )

    def get_and_cache_action_embeddings(self, obs, actions, stop_grad=False):
        self._cache_action_embeddings(self._get_node_embeddings(obs), actions, stop_grad)

    @abstractmethod
    def train(self):
        raise NotImplementedError

    @abstractmethod
    def test(self):
        raise NotImplementedError

    @abstractmethod
    def set_device(self, device):
        raise NotImplementedError

    @abstractmethod
    def get_checkpoint(self):
        raise NotImplementedError

    @abstractmethod
    def load_checkpoint(self, checkpoint):
        raise NotImplementedError

