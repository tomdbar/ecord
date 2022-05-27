import time
from copy import deepcopy

import torch
from torch_scatter import scatter_max, scatter_min

from ecord.solvers.actors.ecord import _ActorBase


def scatter_softmax_sample(x, batch, dim=-1):
    '''
    https://stackoverflow.com/questions/4463561/weighted-random-selection-from-array
    '''
    x_corr = x - scatter_max(x, batch, dim=0)[0][batch]
    z = -(torch.rand_like(x)).log() / x_corr.exp()
    return scatter_min(z, batch, dim=dim)[1]

def maybe_cuda_synchronise():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


class Actor_Q(_ActorBase):

    def __init__(self,
                 gnn,
                 add_glob_to_nodes=True,
                 device=None,
                 *args,
                 **kwargs
                 ):

        self.gnn = gnn
        self.gnn_target = deepcopy(gnn)
        self.add_glob_to_nodes = add_glob_to_nodes

        if device is not None:
            self.set_device(device)

        self.reset_state()

    def parameters(self):
        return self.gnn.parameters()

    def update_target_network(self, polyak=1):
        self.gnn_target = self.__update_network_params(self.gnn, self.gnn_target, polyak)

    def __update_network_params(self, src, trg, polyak=1):
        if polyak == 1:
            trg.load_state_dict(src.state_dict())
            for param in trg.parameters():
                param.requires_grad = False
        else:
            for target_param, param in zip(trg.parameters(), src.parameters()):
                target_param.data.copy_((1 - polyak) * target_param.data + polyak * param.data)
        return trg

    def _get_node_features(self, obs):
        node_features = obs.node_features.to(self.device)
        if obs.glob_features is not None and self.add_glob_to_nodes:
            node_features = torch.cat([node_features, obs.glob_features.to(self.device)[obs.batch_idx]], dim=-1)
        return node_features

    def get_Q_values(self, obs, use_target_network=False, actor_idx=None, mask=None):
        '''Calculate Q-values based on an observation of the environment
        and the current internal state of the actor.
        '''
        if use_target_network:
            gnn = self.gnn_target
        else:
            gnn = self.gnn

        assert (actor_idx in [None, 0]), "Invalid default_actor_idx."

        feats = self._get_node_features(obs)
        preds = gnn(feats, obs.edge_index, obs.edge_attr, obs.batch_idx, obs.degree, out_device=self.device)  # (iii)

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

        if return_timings:
            t = time.time()
            maybe_cuda_synchronise()
        q_vals = self.get_Q_values(obs, use_target_network, mask)
        if return_timings:
            maybe_cuda_synchronise()
            times['calc_q_vals'] = time.time() - t
            t = time.time()

        if tau==0:
            action_vals, actions = scatter_max(q_vals, obs.batch_idx.to(q_vals.device), dim=0)
        else:
            actions = scatter_softmax_sample(q_vals.detach() / tau, obs.batch_idx.to(q_vals.device), dim=0)
        if return_timings:
            maybe_cuda_synchronise()
            times['action_select'] = time.time() - t

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
            # No-op for ECO-DQN.
            pass

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
        self.gnn_target.train()

    def test(self):
        self.gnn.eval()
        self.gnn_target.eval()

    def set_device(self, device):
        self.device = device
        self.gnn.to(device)
        self.gnn_target.to(device)

    def get_checkpoint(self):
        checkpoint = {
            'gnn:state_dict': self.gnn.state_dict(),
            'gnn_target:state_dict': self.gnn.state_dict(),
        }
        return checkpoint

    def load_checkpoint(self, checkpoint):
        self.gnn.load_state_dict(checkpoint['gnn:state_dict'])
        self.gnn_target.load_state_dict(checkpoint['gnn_target:state_dict'])

    ##
    # Extra no-op functions to match ECORD.
    ##
    def reset_state(self):
        pass

    def get_state(self):
        return None

    def set_state(self, *args, **kwargs):
        pass

    def initialise_node_embeddings(self, batch, num_tradj, use_target_network=False):
        pass

    def initialise_embeddings(self, batch, num_tradj, use_target_network=False):
        pass

    def get_node_dist_from_gnn_embeddings(self):
        return None

    def _update_internal_state(self, obs, use_target_network=False):
        pass

    def _cache_action_embeddings(self, embeddings, actions, stop_grad=False):
        pass

    def get_and_cache_action_embeddings(self, obs, actions, stop_grad=False):
        pass