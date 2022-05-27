from collections import deque, defaultdict
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F

from ecord.environment.data import Batch, get_batch_info
from ecord.solvers.actors._base import ActorState
from ecord.environment.environment import EnvObservation

import random


@dataclass
class Transition:
    '''A single training sample for k-BPTT burn in and n-step rollout.
    '''

    batch: Batch
    actor_state_start: ActorState
    node_features: torch.Tensor
    glob_features: torch.Tensor
    action: torch.Tensor
    reward: torch.Tensor
    done: torch.Tensor
    k_BPTT: torch.Tensor
    n_step: torch.Tensor
    best_state: torch.Tensor = None


@dataclass
class TransitionBatch:
    '''A batch of training samples.

    batch: The Batch object for the graph.
    actor_critic_state_start: The actor state at time t-k.
    obs: k+n+1 EnvObservation's, for times t-k,....,t+n.
    action: k+n+1 actions, for times t-k,....,t+n.
    reward: The cumulative discounted reward from time steps t+1...t+n.
    done: The done flag at step t+n.
    k_BPTT: The number of 'burn-in' steps required (may vary across samples in the batch).
    '''

    batch: Batch
    actor_state_start: ActorState
    obs: EnvObservation
    action: torch.Tensor
    reward: torch.Tensor
    done: torch.Tensor
    k_BPTT: torch.Tensor
    n_step: torch.Tensor
    best_state: torch.Tensor = None


class ReplayBuffer:
    """
    Replay Buffer for storing past experiences allowing the agent to learn from them
    Args:
        capacity: size of the buffer
    """

    def __init__(self,
                 graph_batcher,
                 capacity,
                 add_probability=1,
                 k_BPTT=10,
                 n_step=1,
                 gamma=0.99):
        self.graph_batcher = graph_batcher
        self.add_probability = add_probability
        self.buffer = deque(maxlen=capacity)
        self.k_BPTT = k_BPTT
        self.n_step = n_step
        self.gamma = gamma

    def __len__(self) -> None:
        return len(self.buffer)

    def prepare_for_batch(self, batch):
        if hasattr(self, "_tmp_buffers"):
            assert len(self._tmp_buffers)==0, "Can't prepare for new batch until .flush() is called!"

        self._graph_data_list = batch.tg.to_data_list()
        self._num_nodes_list = (batch.tg.ptr[1:] - batch.tg.ptr[:-1]).tolist()

        self._actor_state_q = deque([], self.k_BPTT + self.n_step + 1)
        self._node_features_q = deque([], self.k_BPTT + self.n_step + 1)
        self._glob_features_q = deque([], self.k_BPTT + self.n_step + 1)
        self._action_q = deque([], self.k_BPTT + self.n_step + 1)
        self._reward_q = deque([], self.n_step + 1)
        self._done_q = deque([], self.n_step + 1)

        self._num_graphs = len(self._num_nodes_list)
        self._num_par_samples = None
        self._num_tradj = None
        self._tmp_buffers = defaultdict(list)

    @staticmethod
    def _flatten(t):
        if t is None:
            return None
        else:
            return t.view(-1, t.size(-1))

    def unpack_samples(self, actor_state, obs):
        def unpack_actor_states(actor_state):
            actor_hidden_embeddings = self._flatten(actor_state.hidden_embeddings)
            actor_init_node_embeddings = self._flatten(
                actor_state.init_node_embeddings
            ).split_with_sizes(
                list(np.repeat(self._num_nodes_list, self._num_tradj))
            )
            actor_last_selected_embeddings = self._flatten(actor_state.last_selected_node_embeddings)
            actor_cell_embeddings = self._flatten(actor_state.cell_embeddings)
            actor_graph_embeddings = self._flatten(actor_state.graph_embeddings)

            lab_args = ['hidden_embeddings', 'init_node_embeddings']
            zip_args = [actor_hidden_embeddings, actor_init_node_embeddings]
            if actor_last_selected_embeddings is not None:
                lab_args.append('last_selected_node_embeddings')
                zip_args.append(actor_last_selected_embeddings)
            if actor_cell_embeddings is not None:
                lab_args.append('cell_embeddings')
                zip_args.append(actor_cell_embeddings)
            if actor_graph_embeddings is not None:
                lab_args.append('graph_embeddings')
                zip_args.append(actor_graph_embeddings)

            actor_state = [ActorState(**dict([(k, v) for k, v in zip(lab_args, args)])) for args in zip(*zip_args)]

            return actor_state

        node_features = self._flatten(
            obs.node_features
        ).split_with_sizes(list(np.repeat(self._num_nodes_list, self._num_tradj)))
        glob_features = [x for x in self._flatten(obs.glob_features.cpu())]

        if actor_state is not None:
            actor_states = unpack_actor_states(actor_state.detach().cpu())
        else:
            actor_states = [None for _ in range(len(node_features))]

        return actor_states, node_features, glob_features

    def get_transition_from_queue(self, idx):
        '''
        Generate a transition with all the information required for:
            - initialising the actor at time t-k.
            - a k-step burn in to prepare the actor to time-step t.
            - an n-step rollout to prepare the actor to time-step t+n.
            - the n-step return information for t --> t+n.

        transition = Transition(
                batch: Graph data.
                actor_critic_state_start: actor state at time t-k.
                node_features: k+n node features for steps t-k to t+n,
                glob_features=glob_features: k+n glob features for steps t-k to t+n
                action: actions taken at t-k to t+n (though the action at t+n is not needed for DQN)
                reward: reward received at t to t+n-1
                done: done flag to t to t+n-1
                k_BPTT: number of burn in steps, k
                n_step: number of steps for which to calc the td target, n
            )
        '''
        graph_data = self._graph_data_list[idx // self._num_tradj]
        actor_state_start = self._actor_state_q[0][idx]
        node_features = torch.stack([x[idx] for x in self._node_features_q], dim=0)
        glob_features = torch.stack([x[idx] for x in self._glob_features_q], dim=0)
        action = torch.stack([x[idx] for x in self._action_q], dim=0)
        reward = [x[idx] for x in self._reward_q][:-1]
        done = [x[idx] for x in self._done_q][:-1]

        assert len(reward)==self.n_step, "incorrect number of rewards sampled."
        assert len(done) == self.n_step, "incorrect number of dones sampled."

        if len(done) > 1 and done[-2].bool():
            # If we have (n>1)-step returns and t+n-1 is already done,
            # return None as we don't want this transition.
            return None

        else:
            transition = Transition(
                batch=graph_data,
                actor_state_start=actor_state_start,
                node_features=node_features,
                glob_features=glob_features,
                action=action,
                reward=reward,
                done=done,
                k_BPTT=len(node_features) - self.n_step - 1,
                n_step=self.n_step
            )

            return transition

    def __extend_buffer(self, transitions, prob=None):
        """Extend the replay buffer with transitions."""
        if prob is None:
            prob = self.add_probability
        if prob < 1:
            transitions = [t for t in transitions if random.random() < prob]
        self.buffer.extend(
            [t for t in transitions if t is not None]
        )

    # def append(self, obs, action, reward, done, actor_state):
    def append(self, actor_state, obs, action, reward, done, flush=True):
        """
        Add transition to the buffer:
            actor_state : internal state of actor at time t.
            obs : observation seen by actor at time t.
            action: action at time t, conditioned on actor_state and obs.
            reward: reward at time t, for taking action at time t.
            done: done flag at time t+1.
            flush: whether to immediately add these transitions of store them to later be flushed
                   to the buffer.
        """
        if self._num_tradj is None:
            self._num_tradj = obs.num_tradj
        if self._num_par_samples is None:
            self._num_par_samples = len(reward.view(-1))

        actor_state, node_features, glob_features = self.unpack_samples(actor_state, obs)
        self._actor_state_q.append(actor_state)
        self._node_features_q.append(node_features)
        self._glob_features_q.append(glob_features)
        self._action_q.append(action.view(-1).detach().cpu())
        self._reward_q.append(reward.view(-1).detach().cpu())
        self._done_q.append(done.view(-1).detach().cpu())

        if len(self._actor_state_q) > self.n_step + 1:
            trans = [self.get_transition_from_queue(idx) for idx in range(self._num_par_samples)]
            if flush:
                # immediately add transitions to the buffer
                self.__extend_buffer(trans)
                # self.buffer.extend(
                #     [t for t in trans if t is not None]
                # )
            else:
                # store transitions in per episode buffer
                for idx, t in enumerate(trans):
                    self._tmp_buffers[idx].append(t)

    def flush(self, t_final=None, state_best=None):
        """
        Add transitions stored in temporary buffers to the replay buffer.  Optionally truncate
        these to only the first n steps.

        t_final:
            - None : Add all transitions.
            - int : Add all transitions upto and including t_final.
            - [int1, int2, ...] : A different t_final for each graph (len(t_final)==self._num_graphs)
                                 or tradjectory (len(t_final)==self._num_par_samples)

        state_best:
            - None : Don't add final scores for each episode.
            - [array(...), array(...), ...] : A different final state for each graph (len(state_best)==self._num_graphs)
                                              or tradjectory (len(t_final)==self._num_par_samples)
        """
        if isinstance(t_final, Iterable):
            t_final = np.array(t_final).flatten().astype(int)
            if len(t_final) == self._num_graphs:
                t_final = np.repeat(t_final, self._num_tradj)
            assert len(t_final) == self._num_par_samples, "Unrecognised format for t_final."
        elif not t_final is None:
            t_final = int(t_final)

        if isinstance(state_best, Iterable):
            # scores_final = np.array(state_best).flatten().astype(np.int)
            assert len(state_best) == self._num_par_samples, "Unrecognised format for scores_final."
            def add_state(transition, state):
                transition.batch.best_state = state
                return transition

        for idx, tmp_buffer in self._tmp_buffers.items():
            if state_best is not None:
                state = state_best[idx]
                tmp_buffer = [add_state(t, state) for t in tmp_buffer if t is not None]

            if t_final is None:
                self.__extend_buffer(tmp_buffer)
                # self.buffer.extend([t for t in tmp_buffer if t is not None])
            else:
                if isinstance(t_final, Iterable):
                    t_step = t_final[idx]
                else:
                    t_step = t_final
                self.__extend_buffer(tmp_buffer[:t_step])
                # self.buffer.extend([t for t in tmp_buffer[:t_step] if t is not None])

        self._tmp_buffers = defaultdict(list)

    @torch.no_grad()
    def sample(self, batch_size, out_device='cpu'):
        '''
        Return a batch of transitions in the form required to do BPTT for k steps
        and then calculate n-step returns.

        The total number of time-steps returned will be k_max, which in general will
        be k_BPTT.  However, if none of the sampled transitions can have length
        k_BPTT (i.e. they are too early in the episode), k_max will be the longest
        "burn-in" period of the samples.
        '''
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        transitions = [self.buffer[idx] for idx in indices]

        k_BPTT = torch.tensor([t.k_BPTT for t in transitions])
        n_step = torch.tensor([t.n_step for t in transitions])

        k_max = k_BPTT.max()

        # batch_tg = batch_from_data_list([t.batch for t in transitions])
        batch_tg = self.graph_batcher.batch_tg_from_data_list(
            [t.batch for t in transitions]
        )
        batch_info = get_batch_info(batch_tg.batch, batch_tg.ptr[1:] - batch_tg.ptr[:-1])
        batch = Batch(tg=batch_tg.to(out_device), nx=None, info=batch_info)

        ###
        # actor_critic_state_start : ActorState
        ###

        def _format_embeddings(embeddings, stack=True):
            if embeddings[0] is None:
                embeddings = None
            else:
                if stack:
                    embeddings = torch.stack(embeddings, dim=0).unsqueeze(1).to(out_device)  # unsqueeze --> [graphs, num_tradj=1, dim]
                else:
                    embeddings = torch.cat(embeddings, dim=0).unsqueeze(1).to(out_device)  # unsqueeze --> [nodes in batch, num_tradj=1, dim]
            return embeddings

        if transitions[0].actor_state_start is not None:
            actor_state_start = ActorState(
                hidden_embeddings=_format_embeddings(
                    [t.actor_state_start.hidden_embeddings for t in transitions]
                ),
                init_node_embeddings=_format_embeddings(
                    [t.actor_state_start.init_node_embeddings for t in transitions], stack=False
                ),
                last_selected_node_embeddings=_format_embeddings(
                    [t.actor_state_start.last_selected_node_embeddings for t in transitions]
                ),
                cell_embeddings=_format_embeddings(
                    [t.actor_state_start.cell_embeddings for t in transitions]
                ),
                graph_embeddings=_format_embeddings(
                    [t.actor_state_start.graph_embeddings for t in transitions]
                ),
            )
        else:
            actor_state_start = None

        ###
        # obs_list : [EnvObservation_1, ..., EnvObservation_(max_len)]
        ###
        # [k_max, num_nodes, dim]
        node_features = torch.cat(
            [F.pad(t.node_features, (0, 0, 0, 0, 0, k_max - t.k_BPTT)) for t in transitions], dim=1
        ).unsqueeze(2)  # unsqueeze --> [k_max, num_nodes, num_tradj=1, dim]
        # [k_max, num_samples, dim]
        glob_features = torch.stack(
            [F.pad(t.glob_features, (0, 0, 0, k_max - t.k_BPTT)) for t in transitions], dim=1
        ).unsqueeze(2)  # unsqueeze --> [k_max, num_samples, num_tradj=1, dim]

        obs_list = [
            EnvObservation(
                node_features = node,
                glob_features = glob,
                edge_index = batch.tg.edge_index,
                edge_attr = batch.tg.edge_attr,
                batch_idx = batch.tg.batch,
                batch_ptr = batch.tg.ptr,
                degree = batch.tg.degree,
                num_tradj = 1
            )
            for node, glob in zip(node_features, glob_features)
        ]

        ###
        # action : [max_len, num_samples, num_tradj=1]
        ###
        action = torch.stack([F.pad(t.action, (0, k_max - t.k_BPTT)) for t in transitions], dim=1)
        action += batch.tg.ptr[:-1][None, :]  # offset actions by batch.
        action = action.unsqueeze(-1)

        ###
        # reward : [num_samples, n_step] (Recall reward q is only length n-step).
        # returns : [num_samples]
        ###
        reward = torch.tensor([t.reward for t in transitions])
        reward *= (self.gamma ** torch.arange(n_step[0])[None, :])  # discount future rewards
        returns = reward.sum(1)  # sum

        ###
        # returns : [num_samples]
        ###
        done = torch.tensor([t.done[-1] for t in transitions])

        # ###
        # # best_state : [num_samples, num_nodes] or None
        # ###
        best_state = [t.best_state for t in transitions]
        if None in best_state:
            best_state = None
        else:
            best_state = torch.stack(best_state, dim=0)

        transition_batch = TransitionBatch(
            batch=batch,
            actor_state_start=actor_state_start,
            obs=obs_list,
            action=action,
            reward=returns,
            done=done,
            k_BPTT=k_BPTT,
            n_step=n_step,
            best_state=best_state,
        )

        return transition_batch

