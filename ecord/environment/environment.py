from collections import namedtuple

import torch
from torch_scatter import segment_csr
from collections import deque
import time

from dataclasses import dataclass

from ecord.environment import Tradjectories, NodeObservation, GlobalObservation

@dataclass
class EnvObservation:
    node_features: torch.tensor  # [num_nodes, num_tradj, dim]
    glob_features: torch.tensor # [num_graph, num_tradj, dim]
    edge_index: torch.tensor
    edge_attr: torch.tensor
    batch_idx: torch.tensor  # [num_nodes]
    batch_ptr: torch.tensor  # [num_graph+1]
    degree: torch.tensor
    num_tradj: torch.tensor

    def to(self, device):
        for attr_name in [
            'node_features',  # [num_nodes, num_tradj, dim]
            'glob_features',  # [num_graph, num_tradj, dim]
            'edge_index',
            'edge_attr',
            'batch_idx',  # [num_nodes]
            'batch_ptr',  # [num_graph+1]
            'degree']:
            attr = getattr(self, attr_name)
            if attr is not None:
                setattr(self, attr_name, attr.to(device))

class Environment:

    def __init__(self,
                 batch_nx,
                 num_tradj,
                 batch_tg,
                 node_features=[
                     NodeObservation.STATE,
                     NodeObservation.PEEK_NORM,
                     NodeObservation.STEPS_STATIC_CLIP,
                 ],
                 glob_features=[
                     GlobalObservation.SCORE_FROM_BEST_NORM,
                     GlobalObservation.PEEK_MAX_NORM,
                 ],
                 init_state=None,
                 intermediate_reward=0,
                 revisit_reward=0,
                 add_edges_to_observations=False,
                 device='cpu'):
        self.device = device
        self.tradj = Tradjectories(
            batch_nx,
            num_tradj,
            batch_tg,
            node_features,
            glob_features,
            init_state
        )

        self._batch_idx = batch_tg.batch.to(self.device)
        self._batch_ptr = batch_tg.ptr.to(self.device)
        self._batch_ptr_cpu = batch_tg.ptr.cpu()
        self._num_tradj = num_tradj

        if add_edges_to_observations:
            # If we need the graph structure, store it here.
            self._edge_index = batch_tg.edge_index.to(self.device)
            self._edge_attr = batch_tg.edge_attr.to(self.device)
            self._degree = batch_tg.degree.to(self.device)
        else:
            # If we don't need the graph structure, use None's.
            self._edge_index = None
            self._edge_attr = None
            self._degree = None

        self.scores = self.tradj.get_scores().clone()
        self.best_scores = self.tradj.get_best_scores().clone()
        self.num_nodes = torch.from_numpy(self.tradj._num_nodes_list)[:, None]

        self._intermediate_reward = intermediate_reward
        self._revisit_reward = revisit_reward

        if self._intermediate_reward != 0:
            self.state_last_min = None
        if self._revisit_reward != 0:
            self.state_history = None

    def observe(self):
        # node_features : [nodes, num_tradj, dim]
        node_features = self.tradj.get_node_features(flat=True).to(self.device)
        # glob_features : [graphs, num_tradj, dim]
        glob_features = self.tradj.get_glob_features().to(self.device)

        obs = EnvObservation(
            node_features = node_features,
            glob_features = glob_features,
            edge_index = self._edge_index,
            edge_attr = self._edge_attr,
            batch_idx = self._batch_idx,
            batch_ptr = self._batch_ptr,
            degree = self._degree,
            num_tradj = self._num_tradj
        )

        return obs

    def _calc_reward(self, new_scores):
        reward = (new_scores - self.best_scores).clamp(min=0) / self.num_nodes

        if self._intermediate_reward != 0:
            peeks = self.tradj.get_peeks(flat=True)
            is_local_minima = (segment_csr((peeks > 0).int(), self._batch_ptr_cpu, reduce='sum') == 0)
            is_new_local_minima = is_local_minima.clone()
            state = self.tradj.get_states(flat=True)

            if self.state_last_min is not None:
                for g_idx, t_idx in zip(*torch.where(is_local_minima)):
                    s = state[self._batch_ptr[g_idx]:self._batch_ptr[g_idx + 1], t_idx]
                    if self.state_last_min is not None:
                        s_last = self.state_last_min[self._batch_ptr[g_idx]:self._batch_ptr[g_idx + 1], t_idx]
                        if (s == s_last).all():
                            is_new_local_minima[g_idx, t_idx] = False
                        else:
                            self.state_last_min[self._batch_ptr[g_idx]:self._batch_ptr[g_idx + 1], t_idx] = s

            else:
                self.state_last_min = -1*torch.ones_like(state)

            reward[is_new_local_minima & (reward > 0)] += self._intermediate_reward

        if self._revisit_reward != 0:
            is_revisited_local_minima = is_local_minima & ~is_new_local_minima
            reward[is_revisited_local_minima] += self._revisit_reward

        return reward

    def step(self, actions, ret_reward=True):
        self.tradj.step(actions.detach().cpu(), apply_batch_offset=True)

        new_scores = self.tradj.get_scores().clone()
        if ret_reward:
            reward = self._calc_reward(new_scores)
        else:
            reward = None

        self.scores = new_scores
        self.best_scores = self.tradj.get_best_scores().clone()

        return self.observe(), reward
