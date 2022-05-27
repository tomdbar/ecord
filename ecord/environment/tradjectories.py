import numpy as np
import torch
import torch.nn.functional as F
from torch_scatter import scatter_mean, scatter_std
from torch_scatter import segment_csr, segment_coo, scatter_max

from ecord.environment._tradjectory_step_cy import step_cy
from ecord.environment.observations import NodeObservation, GlobalObservation
from ecord.environment.utils import get_laplacian_PEs


class Tradjectories:
    """Computes and stores multiple parallel optimisation tradjectories on batches of graphs."""

    _lap_PEs_k = 3
    _clip_features_max = 40

    def __init__(self,
                 batch_nx,
                 num_tradj,
                 batch_tg=None,
                 node_features=[NodeObservation.STATE, NodeObservation.PEEK],
                 glob_features=[],
                 init_state=None):

        # A whole bunch of ugly set-up designed to compute the step function as fast as possible.
        batch_nx = [G.to_directed(as_view=True) if not G.is_directed() else G for G in batch_nx]

        self.num_tradj = num_tradj
        self.num_graphs = len(batch_nx)
        self.num_nodes_max = max([G.number_of_nodes() for G in batch_nx])

        # Prepare  all of the internal features needed for fast tradjectory tracking.
        # (These are more focussed on speed than interpretability!)
        self.num_nodes = batch_tg.num_nodes

        self._num_nodes_list = np.array([G.number_of_nodes() for G in batch_nx])
        self._num_nodes_list_expanded = self._num_nodes_list[batch_tg.batch.numpy(), None]
        self._batch_offset = torch.from_numpy((np.roll(np.cumsum(self._num_nodes_list, 0), 1)))
        self._batch_offset[0] = 0
        self._edges_and_attrs_by_node = torch.cat(batch_tg.edges_and_attrs_by_node, 1).numpy().astype(np.float32)

        if any(self._num_nodes_list < self.num_nodes_max):
            self._flat2batch_idxs = torch.stack(
                [batch_tg.batch, torch.cat([torch.arange(n) for n in self._num_nodes_list])])
        else:
            self._flat2batch_idxs = None

        self._edges_by_node = torch.cat(batch_tg.edges_by_node, 1).numpy().astype(np.int32)
        self._edge_attrs_by_node = torch.cat(batch_tg.edge_attrs_by_node).numpy().astype(np.float32)

        degree = batch_tg.degree.float()
        self.degree = batch_tg.degree.numpy().astype(np.int32)
        self.degree_norm = (degree - scatter_mean(degree, batch_tg.batch)[batch_tg.batch]) / scatter_std(degree, batch_tg.batch)[batch_tg.batch]
        self.degree_norm = self.degree_norm.numpy().astype(np.float32)

        self._batch = batch_tg.batch

        self._edges_and_attrs_by_node_ptr = np.cumsum(self.degree, 0).astype(np.int32)
        self._edges_and_attrs_by_node_ptr = np.pad(self._edges_and_attrs_by_node_ptr, (1, 0))

        # Prepare tradjectory statistics.
        if init_state is None:
            states = np.random.randint(0, 2, (self.num_nodes, self.num_tradj), dtype=np.int8)
        else:
            assert init_state.shape == (self.num_nodes, self.num_tradj), \
                f"init_state must have shape {(self.num_nodes, self.num_tradj)}.  Got {init_state.shape}."
            if torch.is_tensor(init_state):
                init_state = init_state.cpu().numpy()
            states = init_state.astype(np.int8)

        # Initialise state and corresponding scores/peeks.
        self.__w_ij = batch_tg.edge_attr
        self.__e_ij = batch_tg.edge_index
        self.__edge_ptr = torch.tensor([0] + [G.number_of_edges() for G in batch_nx]).cumsum(0)
        self.__degree_ptr = F.pad(batch_tg.degree.cumsum(0), (1, 0), value=0)

        scores, peeks = self._calc_scores_and_peeks(states)

        self._scores = scores
        self._best_scores = scores.copy()

        self._node_features = node_features.copy()
        self._glob_features = glob_features.copy()

        self.num_node_features = len(self._node_features)
        self.num_glob_features = len(self._glob_features)

        if (
                (NodeObservation.PEEK not in self._node_features)
                and (NodeObservation.PEEK_NORM not in self._node_features)
                and (NodeObservation.PEEK_SCALE not in self._node_features)
        ):
            self._node_features.append(NodeObservation.PEEK)
            self.hide_peek = True
        else:
            self.hide_peek = False

        self._node_features_idxs, node_features_norm_mask, self._node_features_clip_mask, self._node_features_scale_mask = self.__get_node_features_idxs(self._node_features)
        self._glob_features_idxs, glob_features_norm_mask, self._glob_features_clip_mask, self._glob_features_scale_mask = self.__get_glob_features_idxs(self._glob_features)

        self._dim_node_features = len(self._node_features)
        if NodeObservation.LAPLACIAN_PE in self._node_features:
            self._dim_node_features += (self._lap_PEs_k - 1)
            self.lap_PEs = get_laplacian_PEs(batch_nx, k=self._lap_PEs_k, num_tradj=self.num_tradj, randomise_signs=True)
        else:
            self.lap_PEs = None
        self._dim_glob_features = len(self._glob_features)

        self._node_features = self._init_node_features(states, peeks)
        self._glob_features = self.__init_glob_features(scores, peeks, batch_tg)

        self._node_features_norm = np.ones_like(self._node_features)
        self._glob_features_norm = np.ones_like(self._glob_features)

        for idx in np.where(node_features_norm_mask)[0]:
            self._node_features_norm[..., idx] = self._num_nodes_list_expanded
        for idx in np.where(glob_features_norm_mask)[0]:
            self._glob_features_norm[..., idx] = self._num_nodes_list[:, None]

        self._best_states = self.get_states(flat=True).numpy().copy()

        max_local_rewards_available = self.get_peeks().max(dim=1).values

        self._max_local_rewards_available = {
            'over_graphs': max_local_rewards_available.unsqueeze(-1).numpy(),
            'over_nodes' : max_local_rewards_available.unsqueeze(-1)[batch_tg.batch].numpy()
        }

    def _calc_scores_and_peeks(self, s):
        if isinstance(s, np.ndarray):
            s = torch.from_numpy(s)

        s_i, s_j = s[self.__e_ij]

        scores = 0.5 * segment_csr(
            self.__w_ij * (s_i + s_j).fmod(2),
            self.__edge_ptr,
            reduce='sum'
        ).numpy().astype(np.float32)

        peeks = segment_csr(
            self.__w_ij * (2 * s_i - 1) * (2 * s_j - 1),
            self.__degree_ptr,
            reduce='sum'
        ).numpy().astype(np.float32)

        return scores, peeks

    def _init_node_features(self, states, peeks):
        node_features = np.zeros((self.num_nodes, self.num_tradj, self._dim_node_features), dtype=np.float32)
        if self._node_features_idxs[0] >= 0:  # state
            node_features[..., self._node_features_idxs[0]] = states.copy()
        if self._node_features_idxs[1] >= 0:  # peek
            node_features[..., self._node_features_idxs[1]] = peeks.copy()
        if self._node_features_idxs[2] >= 0:  # best_state
            node_features[..., self._node_features_idxs[2]] = states.copy()
        if self._node_features_idxs[3] >= 0:  # steps_static
            node_features[..., self._node_features_idxs[3]] = 0
        if self._node_features_idxs[4] >= 0:  # degree
            node_features[..., self._node_features_idxs[4]] = self.degree_norm.copy()[:,None]
        if NodeObservation.LAPLACIAN_PE in self._node_features:
            node_features[..., -self._lap_PEs_k:] = self.lap_PEs.view(*node_features[..., -self._lap_PEs_k:].shape)
        if NodeObservation.ID in self._node_features:
            node_features[..., -1] = np.linspace(0,1,self.num_nodes)[:,None].repeat(self.num_tradj,1)
        return node_features

    def __init_glob_features(self, scores, peeks, batch_tg):
        if all(self._glob_features_idxs) < 0:
            glob_features = None
        else:
            glob_features = np.zeros((self.num_graphs, self.num_tradj, self._dim_glob_features), dtype=np.float32)
            if self._glob_features_idxs[0] >= 0:  # score_from_best
                glob_features[..., self._glob_features_idxs[0]] = 0
            if self._glob_features_idxs[1] >= 0:  # steps_since_best
                glob_features[..., self._glob_features_idxs[1]] = 0
            if self._glob_features_idxs[2] >= 0:  # num_greedy
                num_greedy = segment_coo((torch.from_numpy(peeks.copy()) > 0).int(), batch_tg.batch, reduce='sum').numpy()
                glob_features[..., self._glob_features_idxs[2]] = num_greedy
            if self._glob_features_idxs[3] >= 0:  # steps
                glob_features[..., self._glob_features_idxs[3]] = 0
            if self._glob_features_idxs[4] >= 0:  # max_peeks
                glob_features[..., self._glob_features_idxs[4]] = self.get_peeks().max(1).values
            if self._glob_features_idxs[5] >= 0:  # mean_peeks
                glob_features[..., self._glob_features_idxs[4]] = self.get_peeks().mean(1)

        return glob_features

    def __get_node_features_idxs(self, node_features):
        _node_features_idxs = -1 * np.ones(5, dtype=np.intc)  # [state, peek, best_state, steps_static, degree]
        feat_dim = len(node_features)
        if NodeObservation.LAPLACIAN_PE in self._node_features:
            feat_dim += self._lap_PEs_k - 1
        _node_features_norm = np.zeros(feat_dim, dtype=bool)
        _node_features_clip = np.zeros(feat_dim, dtype=bool)
        _node_features_scale = np.zeros(feat_dim, dtype=bool)
        _state_configured, _peek_configured, _static_configured, _degree_configured = False, False, False, False

        for idx, obs in enumerate(node_features):

            if obs is NodeObservation.STATE:
                _node_features_idxs[0] = idx
                _state_configured = True

            elif obs in [NodeObservation.PEEK, NodeObservation.PEEK_NORM, NodeObservation.PEEK_SCALE]:
                if not _peek_configured:
                    _node_features_idxs[1] = idx
                    _peek_configured = True
                    if obs is NodeObservation.PEEK_NORM:
                        _node_features_norm[idx] = True
                    elif obs is NodeObservation.PEEK_SCALE:
                        _node_features_scale[idx] = True
                else:
                    raise Exception("Only one of NodeObservation.PEEK/NodeObservation.PEEK_NORM/NodeObservation.PEEK_SCALE should be configured.")

            elif obs is NodeObservation.STATE_BEST:
                _node_features_idxs[2] = idx

            elif obs in [NodeObservation.STEPS_STATIC, NodeObservation.STEPS_STATIC_NORM, NodeObservation.STEPS_STATIC_CLIP]:
                if not _static_configured:
                    _node_features_idxs[3] = idx
                    _static_configured = True
                    if obs is NodeObservation.STEPS_STATIC_NORM:
                        _node_features_norm[idx] = True
                    elif obs is NodeObservation.STEPS_STATIC_CLIP:
                        _node_features_clip[idx] = True
                else:
                    raise Exception(
                        "Only one of NodeObservation.STEPS_STATIC/NodeObservation.STEPS_STATIC_NORM should be configured.")

            elif obs is NodeObservation.DEGREE_NORM:
                _node_features_idxs[4] = idx
                _degree_configured = True

        assert _state_configured, "NodeObservation.STATE must be configured."
        assert _peek_configured, "Exactly one of NodeObservation.PEEK/NodeObservation.PEEK_NORM/NodeObservation.PEEK_SCALE must be configured."

        return _node_features_idxs, _node_features_norm, _node_features_clip, _node_features_scale

    def __get_glob_features_idxs(self, glob_features):
        _glob_features_idxs = -1 * np.ones(6, dtype=np.intc)  # [score_from_best, steps_since_best, num_greedy, steps, max_peek, mean_peek]
        _glob_features_norm = np.zeros(len(glob_features), dtype=bool)
        _glob_features_clip = np.zeros(len(glob_features), dtype=bool)
        _glob_features_scale = np.zeros(len(glob_features), dtype=bool)
        _score_from_best_configured = False
        _score_since_best_configured = False
        _num_greedy_configured = False
        _steps_configured = False
        _peek_max_configured = False
        _peek_mean_configured = False

        for idx, obs in enumerate(glob_features):

            if obs in [GlobalObservation.SCORE_FROM_BEST, GlobalObservation.SCORE_FROM_BEST_NORM, GlobalObservation.SCORE_FROM_BEST_SCALE]:
                if not _score_from_best_configured:
                    _glob_features_idxs[0] = idx
                    _score_from_best_configured = True
                    if obs is GlobalObservation.SCORE_FROM_BEST_NORM:
                        _glob_features_norm[idx] = True
                    elif obs is GlobalObservation.SCORE_FROM_BEST_SCALE:
                        _glob_features_scale[idx] = True
                else:
                    raise Exception(
                        "Only one of GlobalObservation.SCORE_FROM_BEST/GlobalObservation.SCORE_FROM_BEST_NORMshould be configured.")

            elif obs in [GlobalObservation.STEPS_SINCE_BEST, GlobalObservation.STEPS_SINCE_BEST_NORM, GlobalObservation.STEPS_SINCE_BEST_CLIP]:
                if not _score_since_best_configured:
                    _glob_features_idxs[1] = idx
                    _score_since_best_configured = True
                    if obs is GlobalObservation.STEPS_SINCE_BEST_NORM:
                        _glob_features_norm[idx] = True
                    elif obs is GlobalObservation.STEPS_SINCE_BEST_CLIP:
                        _glob_features_clip[idx] = True
                else:
                    raise Exception(
                        "Only one of GlobalObservation.STEPS_SINCE_BEST, GlobalObservation.STEPS_SINCE_BEST_NORM should be configured.")

            elif obs in [GlobalObservation.NUM_GREEDY_ACTIONS, GlobalObservation.NUM_GREEDY_ACTIONS_NORM, GlobalObservation.NUM_GREEDY_ACTIONS_CLIP]:
                if not _num_greedy_configured:
                    _glob_features_idxs[2] = idx
                    _num_greedy_configured = True
                    if obs is GlobalObservation.NUM_GREEDY_ACTIONS_NORM:
                        _glob_features_norm[idx] = True
                    elif obs is GlobalObservation.NUM_GREEDY_ACTIONS_CLIP:
                        _glob_features_clip[idx] = True

            elif obs in [GlobalObservation.STEPS, GlobalObservation.STEPS_NORM]:
                if not _num_greedy_configured:
                    _glob_features_idxs[3] = idx
                    _steps_configured = True
                    if obs is GlobalObservation.STEPS_NORM:
                        _glob_features_norm[idx] = True

            elif obs is [GlobalObservation.PEEK_MAX, GlobalObservation.PEEK_MAX_NORM, GlobalObservation.PEEK_MAX_SCALE]:
                if not _peek_max_configured:
                    _glob_features_idxs[4] = idx
                    _peek_max_configured = True
                if obs is GlobalObservation.PEEK_MAX_NORM:
                    _glob_features_norm[idx] = True
                if obs is GlobalObservation.PEEK_MAX_SCALE:
                    _glob_features_scale[idx] = True

            elif obs is [GlobalObservation.PEEK_MEAN, GlobalObservation.PEEK_MEAN_NORM]:
                if not _peek_mean_configured:
                    _glob_features_idxs[5] = idx
                    _peek_mean_configured = True
                if obs is GlobalObservation.PEEK_MEAN_NORM:
                    _glob_features_norm[idx] = True

        return _glob_features_idxs, _glob_features_norm, _glob_features_clip, _glob_features_scale

    def get_scores(self):
        return torch.from_numpy(self._scores.copy())

    def get_best_scores(self):
        return torch.from_numpy(self._best_scores.copy())

    def get_states(self, flat=False):
        states = torch.from_numpy(self._node_features[..., self._node_features_idxs[0]].copy())
        if not flat:
            states = self._flat2batch(states, fill_val=-1)
        return states

    def get_best_states(self, flat=False):
        best_states = torch.from_numpy(self._best_states.copy())
        if not flat:
            best_states = self._flat2batch(best_states, fill_val=-1)
        return best_states

    def get_peeks(self, flat=False):
        peeks = torch.from_numpy(self._node_features[..., self._node_features_idxs[1]].copy())
        if not flat:
            peeks = self._flat2batch(peeks, fill_val=-1e10)
        return peeks

    def get_node_features(self, flat=True):
        node_features = self._node_features.copy() / self._node_features_norm
        node_features[..., self._node_features_clip_mask] = np.clip(
            node_features[..., self._node_features_clip_mask], a_min=None, a_max=self._clip_features_max
        ) / self._clip_features_max

        node_features[..., self._node_features_scale_mask] /= self._max_local_rewards_available['over_nodes']

        node_features = torch.from_numpy(node_features)

        if self.hide_peek:
            node_features = node_features[..., :-1]
        if flat:
            node_features = node_features.view(-1, self.num_tradj, node_features.size(-1))
        else:
            node_features = node_features.view(self.num_graphs, self.num_nodes_max, self.num_tradj, -1)

        return node_features

    def get_glob_features(self):
        if self._glob_features is not None:

            glob_features = self._glob_features.copy()

            if self._glob_features_idxs[4] >= 0 or self._glob_features_idxs[5] >= 0:
                peeks = self.get_peeks()
                if self._glob_features_idxs[4] >= 0:
                    glob_features[..., self._glob_features_idxs[4]] = peeks.max(1).values
                if self._glob_features_idxs[5] >= 0:
                    glob_features[..., self._glob_features_idxs[5]] = peeks.mean(1)

            glob_features = glob_features / self._glob_features_norm

            glob_features[..., self._glob_features_clip_mask] = np.clip(
                glob_features[..., self._glob_features_clip_mask], a_min=None, a_max=self._clip_features_max
            ) / self._clip_features_max

            glob_features[..., self._glob_features_clip_mask] /= self._max_local_rewards_available['over_graphs']

            glob_features = torch.from_numpy(glob_features)
        return glob_features

    def _flat2batch(self, arr, fill_val=0):
        if self._flat2batch_idxs is None:
            # Batch of graphs with equal size.
            _arr = arr.view(self.num_graphs, self.num_nodes_max, -1)
        else:
            _arr = fill_val * torch.ones(self.num_graphs, self.num_nodes_max, self.num_tradj)
            _arr[self._flat2batch_idxs[0], self._flat2batch_idxs[1]] = arr
        return _arr

    def step(self, actions, apply_batch_offset=True):
        actions = actions.clone()
        if apply_batch_offset:
            actions = actions + self._batch_offset.unsqueeze(1)

        step_cy(actions.numpy(),
                self._scores,
                self._best_scores,
                self._best_states,
                self._node_features,
                self._glob_features,
                self._node_features_idxs,
                self._glob_features_idxs,
                self._edges_by_node,
                self._edge_attrs_by_node,
                self._edges_and_attrs_by_node_ptr,
                self.num_nodes,
                self.num_graphs,
                self.num_tradj)