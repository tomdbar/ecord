import itertools
import time
from copy import deepcopy
from dataclasses import asdict

import torch
from torch_scatter import scatter_max, scatter_min

from ecord.solvers.actors._base import ActorState, _ActorBase


def scatter_softmax_sample(x, batch, dim=-1):
    '''
    https://stackoverflow.com/questions/4463561/weighted-random-selection-from-array
    '''
    x_corr = x - scatter_max(x, batch, dim=0)[0][batch]
    # x_corr = x - segment_coo(x, batch, reduce='max')[batch]
    z = -(torch.rand_like(x)).log() / x_corr.exp()
    return scatter_min(z, batch, dim=dim)[1]

def maybe_cuda_synchronise():
    if torch.cuda.is_available():
        torch.cuda.synchronize()

class Actor_Q(_ActorBase):
    mask_val = -1e3

    def __init__(
            self,
            gnn,
            rnn,
            gnn2rnn=None,
            node_encoder=None,
            node_classifier=None,
            add_glob_to_nodes=False,
            add_glob_to_obs=False,
            default_actor_idx=None,
            detach_gnn_from_rnn=False,
            device=None,
    ):
        self.gnn_target = deepcopy(gnn)
        self.rnn_target = deepcopy(rnn)
        if node_encoder is not None:
            self.node_encoder_target = deepcopy(node_encoder)
        else:
            self.node_encoder_target = node_encoder

        assert (default_actor_idx in [None, 0]), "Invalid default_actor_idx."

        super().__init__(
            gnn,
            rnn,
            gnn2rnn=gnn2rnn,
            node_encoder=node_encoder,
            node_classifier=node_classifier,
            add_glob_to_nodes=add_glob_to_nodes,
            add_glob_to_obs=add_glob_to_obs,
            detach_gnn_from_rnn=detach_gnn_from_rnn,
            device=device
        )

    def update_target_network(self, polyak=1):
        self.gnn_target = self.__update_network_params(self.gnn, self.gnn_target, polyak)
        self.rnn_target = self.__update_network_params(self.rnn, self.rnn_target, polyak)
        if self.node_encoder_target is not None:
            self.node_encoder_target = self.__update_network_params(self.node_encoder, self.node_encoder_target, polyak)

    def __update_network_params(self, src, trg, polyak=1):
        if polyak == 1:
            trg.load_state_dict(src.state_dict())
            for param in trg.parameters():
                param.requires_grad = False
        else:
            for target_param, param in zip(trg.parameters(), src.parameters()):
                target_param.data.copy_((1 - polyak) * target_param.data + polyak * param.data)
        return trg

    def _get_node_features(self, obs, use_target_network=False):
        node_features = obs.node_features.to(self.device)
        if obs.glob_features is not None and self.add_glob_to_nodes:
            node_features = torch.cat([node_features, obs.glob_features.to(self.device)[obs.batch_idx]], dim=-1)
        if self.node_encoder is not None:
            if not use_target_network:
                node_encoder = self.node_encoder
            else:
                node_encoder = self.node_encoder_target
            # print("node_features (pre)", node_features.shape, "\n", node_features)
            node_features = node_encoder(node_features)
            # print("node_features (post)", node_features.shape, "\n", node_features)
            # raise Exception
        return node_features

    def get_Q_values(self, obs, use_target_network=False, actor_idx=None, mask=None):
        '''Calculate Q-values based on an observation of the environment
        and the current internal state of the actor.

        Given an observation, this requires:
            (i) Update the RNN internal state according to the node-embeddings
                of the last actions taken (at t-1) and the newly observe state of
                the graph (at t).
            (ii) Generate new node embeddings from the GNN embeddings (t=0) and newly
                 observed node features (at t).
            (iii) Calculate Q-values from RNN per-graph state and the per-node embeddings.
        '''
        if use_target_network:
            rnn = self.rnn_target
        else:
            rnn = self.rnn

        assert (actor_idx in [None, 0]), "Invalid default_actor_idx."

        self._update_internal_state(obs, use_target_network)  # (i)

        embeddings = self._get_node_embeddings(obs)  # (ii)

        preds = rnn(embeddings, obs.batch_idx, out_device=self.device)  # (iii)

        if mask is not None:
            preds.masked_fill_(mask, self.mask_val)

        return preds

    def action_select(
            self,
            obs,
            epsilon=0,
            tau=0,
            use_target_network=False,
            cache_selected_embeddings=True,
            return_q_values=False,
            return_timings=False,
            actor_idx=None,
            mask=None
    ):
        '''Select the next action.

        First calculates the Q-values, then selects with epsilon-greedy policy.

        Additionally, stores the embeddings of the selected actions internally,
        for use at the next time-step.
        '''
        assert (actor_idx in [None, 0]), "Invalid default_actor_idx."
        times = {}

        obs.to(self.device)

        if return_timings:
            t = time.time()
            maybe_cuda_synchronise()
        q_vals = self.get_Q_values(obs, use_target_network, mask)
        if return_timings:
            maybe_cuda_synchronise()
            times['calc_q_vals'] = time.time() - t

        if tau==0:
            if return_timings:
                t = time.time()
                maybe_cuda_synchronise()
            action_vals, actions = scatter_max(q_vals, obs.batch_idx, dim=0)
            if return_timings:
                maybe_cuda_synchronise()
                times['action_select'] = time.time() - t
        else:
            actions = scatter_softmax_sample(q_vals.detach() / tau, obs.batch_idx, dim=0)

        # take away batch offset.
        actions -= obs.batch_ptr[:-1][:, None].to(actions.device)
        # times['action_select'] = time.time() - t
        actions = actions.to(self.device)

        if epsilon > 0:
            act_randomly = (torch.rand(*actions.shape) < epsilon)
            if act_randomly.any():
                if mask is None:
                    num_nodes = (obs.batch_ptr[1:] - obs.batch_ptr[:-1]).to(self.device)
                    rand_acts = (torch.rand(actions.shape, device=self.device) * num_nodes[:, None]).long()
                    actions[act_randomly] = rand_acts[act_randomly].to(actions)
                else:
                    # compute random choice from allowed actions as per-mask
                    raise NotImplementedError()

        if cache_selected_embeddings:
            self.get_and_cache_action_embeddings(obs, actions, stop_grad=True)

        ret = [actions]
        if return_q_values:
            ret += [q_vals]
        if return_timings:
            ret += [times]
        if len(ret)==0:
            ret = ret[0]

        return ret

    def train(self):
        self.gnn.train()
        self.rnn.train()
        self.gnn_target.train()
        self.rnn_target.train()
        if self.node_encoder is not None:
            self.node_encoder.train()
        if self.node_encoder_target is not None:
            self.node_encoder_target.train()

    def test(self):
        self.gnn.eval()
        self.rnn.eval()
        self.gnn_target.eval()
        self.rnn_target.eval()
        if self.gnn2rnn is not None:
            self.gnn2rnn.eval()
        if self.node_encoder is not None:
            self.node_encoder.eval()
        if self.node_encoder_target is not None:
            self.node_encoder_target.eval()
        if self.node_classifier is not None:
            self.node_classifier.eval()

    def set_device(self, device):
        self.device = device
        self.gnn.to(device)
        self.rnn.to(device)
        self.gnn_target.to(device)
        self.rnn_target.to(device)
        if self.gnn2rnn is not None:
            self.gnn2rnn.to(device)
        if self.node_encoder is not None:
            self.node_encoder.to(device)
        if self.node_encoder_target is not None:
            self.node_encoder_target.to(device)
        if self.node_classifier is not None:
            self.node_classifier.to(device)

    def get_checkpoint(self):
        checkpoint = {
            'gnn:state_dict': self.gnn.state_dict(),
            'gnn_target:state_dict': self.gnn.state_dict(),
            'rnn:state_dict': self.rnn.state_dict(),
            'rnn_target:state_dict': self.rnn_target.state_dict(),
            'rnn:internal_state': self.rnn.get_internal_state(),
            'init_node_embeddings': self.init_node_embeddings,
            '_last_selected_node_embeddings': self._last_selected_node_embeddings
        }
        if self.gnn2rnn is not None:
            checkpoint['gnn2rnn:state_dict'] = self.gnn2rnn.state_dict()
        if self.node_encoder is not None:
            checkpoint['node_encoder:state_dict'] = self.node_encoder.state_dict()
        if self.node_encoder_target is not None:
            checkpoint['node_encoder_target:state_dict'] = self.node_encoder_target.state_dict()
        if self.node_classifier is not None:
            checkpoint['node_classifier:state_dict'] = self.node_classifier.state_dict()
        return checkpoint

    def load_checkpoint(self, checkpoint):

        def load_maybe(key, target):
            state = checkpoint.get(key, None)
            if (state is not None) and (target is not None):
                target.load_state_dict(state)

        load_maybe('gnn:state_dict', self.gnn)
        load_maybe('gnn_target:state_dict', self.gnn_target)
        load_maybe('rnn:state_dict', self.rnn)
        load_maybe('rnn_target:state_dict', self.rnn_target)

        load_maybe('gnn2rnn:state_dict', self.gnn2rnn)
        load_maybe('node_encoder:state_dict', self.node_encoder)
        load_maybe('node_encoder_target:state_dict', self.node_encoder_target)
        load_maybe('node_classifier:state_dict', self.node_classifier)

        rnn_internal_state = checkpoint.get('rnn:internal_state', None)
        if rnn_internal_state is not None:
            self.rnn.set_internal_state(**checkpoint['rnn:internal_state'])

        rnn_internal_state = checkpoint.get('rnn:internal_state', None)
        if rnn_internal_state is not None:
            self.rnn.set_internal_state(**checkpoint['rnn:internal_state'])

        init_node_embeddings = checkpoint.get('init_node_embeddings', None)
        if init_node_embeddings is not None:
            self.init_node_embeddings = init_node_embeddings

        last_selected_node_embeddings = checkpoint.get('_last_selected_node_embeddings', None)
        if last_selected_node_embeddings is not None:
            self._last_selected_node_embeddings = last_selected_node_embeddings


class Actor_DoubleQ(_ActorBase):
    mask_val = -1e3

    def __init__(
            self,
            gnn,
            rnn,
            node_encoder=None,
            node_classifier=None,
            add_glob_to_nodes=False,
            add_glob_to_obs=False,
            default_actor_idx=None,
            detach_gnn_from_rnn=False,
            device=None
    ):
        self._actor1 = Actor_Q(
            gnn = gnn,
            rnn = rnn,
            node_encoder = node_encoder,
            node_classifier = node_classifier,
            add_glob_to_nodes = add_glob_to_nodes,
            add_glob_to_obs = add_glob_to_obs,
            detach_gnn_from_rnn = detach_gnn_from_rnn,
            device = device,
        )

        def reset(m):
            try:
                m.reset_parameters()
            except:
                pass

        gnn2 = deepcopy(gnn).apply(reset)
        rnn2 = deepcopy(rnn).apply(reset)
        if node_encoder is not None:
            node_encoder2 = deepcopy(node_encoder).apply(reset)
        else:
            node_encoder2 = node_encoder

        self._actor2 = Actor_Q(
            gnn = gnn2,
            rnn = rnn2,
            node_encoder = node_encoder2,
            add_glob_to_nodes = add_glob_to_nodes,
            add_glob_to_obs = add_glob_to_obs,
            detach_gnn_from_rnn = detach_gnn_from_rnn,
            device = device,
        )

        self.default_actor_idx = default_actor_idx

        self.set_device(device)

    def parameters(self):
        return itertools.chain(self._actor1.parameters(), self._actor2.parameters())

    def update_target_network(self, polyak=1):
        self._actor1.update_target_network(polyak)
        self._actor2.update_target_network(polyak)

    def get_Q_values(self, obs, use_target_network=False, mask=None, actor_idx=-1):
        '''Calculate Q-values based on an observation of the environment
        and the current internal state of the actor.

        Given an observation, this requires:
            (i) Update the RNN internal state according to the node-embeddings
                of the last actions taken (at t-1) and the newly observe state of
                the graph (at t).
            (ii) Generate new node embeddings from the GNN embeddings (t=0) and newly
                 observed node features (at t).
            (iii) Calculate Q-values from RNN per-graph state and the per-node embeddings.
        '''
        if actor_idx == -1:
            actor_idx = self.default_actor_idx
        actors = [self._actor1, self._actor2]
        if actor_idx is None:
            preds = [
                actor.get_Q_values(
                    obs=obs,
                    use_target_network=use_target_network,
                    mask=mask,
                )
                for actor in actors
            ]
        else:
            preds = actors[actor_idx].get_Q_values(
                obs=obs,
                use_target_network=use_target_network,
                mask=mask,
            )
        return preds

    def action_select(
            self,
            obs,
            epsilon=0,
            tau=0,
            use_target_network=False,
            cache_selected_embeddings=True,
            return_q_values=False,
            mask=None,
            actor_idx=-1,
    ):
        '''Select the next action.

        First calculates the Q-values, then selects with epsilon-greedy policy.

        Additionally, stores the embeddings of the selected actions internally,
        for use at the next time-step.
        '''
        if actor_idx == -1:
            actor_idx = self.default_actor_idx
        q_vals = self.get_Q_values(obs, use_target_network, mask, actor_idx=actor_idx)
        if actor_idx is None:
            preds = torch.stack(q_vals, dim=-1).min(-1).values
        else:
            preds = q_vals

        if tau is None or tau==0:
            action_vals, actions = scatter_max(preds, obs.batch_idx.to(preds.device), dim=0)
        else:
            actions = scatter_softmax_sample(preds / tau, obs.batch_idx.to(preds.device), dim=0)

        # take away batch offset.
        actions -= obs.batch_ptr[:-1][:, None].to(actions.device)

        if epsilon > 0:
            act_randomly = (torch.rand(*actions.shape) < epsilon)
            if act_randomly.any():
                if mask is None:
                    num_nodes = (obs.batch_ptr[1:] - obs.batch_ptr[:-1]).to(self.device)
                    rand_acts = (torch.rand(actions.shape, device=self.device) * num_nodes[:, None]).long()
                    actions[act_randomly] = rand_acts[act_randomly].to(actions)
                else:
                    # compute random choice from allowed actions as per-mask
                    raise NotImplementedError()

        if cache_selected_embeddings:
            self.get_and_cache_action_embeddings(obs, actions, stop_grad=True)

        if return_q_values:
            return actions, q_vals
        else:
            return actions

    def reset_state(self):
        self._actor1.reset_state()
        self._actor2.reset_state()

    def get_state(self):
        actor_state1 = self._actor1.get_state()
        actor_state2 = self._actor2.get_state()

        state = {}
        for k in asdict(actor_state1).keys():
            attrs = [getattr(s, k) for s in [actor_state1, actor_state2] if getattr(s, k) is not None]
            if len(attrs) > 0:
                state[k] = torch.cat(attrs, dim=-1)
        ActorState(**state)

        return ActorState(**state)

    def set_state(self, actor_state, use_target_network=False, overwite_init_node_embeddings=False, reset_internal_state=False):
        # print(actor_state)
        state1, state2 = {}, {}
        for k in asdict(actor_state).keys():
            attr = getattr(actor_state, k)
            if attr is not None:
                state1[k], state2[k] = attr.chunk(2, dim=-1)

        actor_state1, actor_state2 = ActorState(**state1), ActorState(**state2)

        self._actor1.set_state(
            actor_state1,
            use_target_network=use_target_network,
            overwite_init_node_embeddings=overwite_init_node_embeddings,
            reset_internal_state=reset_internal_state,
        )
        self._actor2.set_state(
            actor_state2,
            use_target_network=use_target_network,
            overwite_init_node_embeddings=overwite_init_node_embeddings,
            reset_internal_state=reset_internal_state,
        )

    def initialise_node_embeddings(self, *args, **kwargs):
        self._actor1.initialise_node_embeddings(*args, **kwargs)
        self._actor2.initialise_node_embeddings(*args, **kwargs)

    def initialise_embeddings(self, *args, **kwargs):
        self._actor1.initialise_embeddings(*args, **kwargs)
        self._actor2.initialise_embeddings(*args, **kwargs)

    def _update_internal_state(self, *args, **kwargs):
        self._actor1._update_internal_state(*args, **kwargs)
        self._actor2._update_internal_state(*args, **kwargs)

    def get_and_cache_action_embeddings(self, *args, **kwargs):
        self._actor1.get_and_cache_action_embeddings(*args, **kwargs)
        self._actor2.get_and_cache_action_embeddings(*args, **kwargs)

    def train(self):
        self._actor1.train()
        self._actor2.train()

    def test(self):
        self._actor1.test()
        self._actor2.test()

    def set_device(self, device):
        self.device = device
        self._actor1.set_device(self.device)
        self._actor2.set_device(self.device)

    def get_checkpoint(self):
        checkpoint = {
            'chk:actor1': self._actor1.get_checkpoint(),
            'chk:actor2': self._actor2.get_checkpoint(),
        }
        return checkpoint

    def load_checkpoint(self, checkpoint):
        self._actor1.load_checkpoint(checkpoint['chk:actor1'])
        self._actor2.load_checkpoint(checkpoint['chk:actor2'])