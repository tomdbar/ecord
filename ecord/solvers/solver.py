import os
import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Multinomial
from torch_scatter import scatter_sum, scatter_max, scatter_mean, scatter_min

from ecord.environment import NodeObservation
from ecord.solvers._base import NetworkSolver
from ecord.utils import mk_dir
from ecord.solvers.actors.ecord import Actor_DoubleQ, Actor_Q
from ecord.solvers.actors.ecodqn import Actor_Q as Actor_Q_ECODQN
from ecord.environment.environment import Environment


def soft_greedy_action_select(peeks, tau=0):
    print(peeks.shape, peeks.type)
    if tau == 0:
        peeks, actions = peeks.max(1)
    else:
        soft_peeks = F.softmax(peeks / tau, 1)
        dist = Multinomial(1, probs=soft_peeks.permute(0, 2, 1))
        actions = dist.sample().argmax(-1)
        peeks = torch.gather(peeks, 1, actions.unsqueeze(1)).squeeze(1)
    return peeks, actions

def scatter_softmax_sample(x, batch, dim=-1):
    '''
    https://stackoverflow.com/questions/4463561/weighted-random-selection-from-array
    '''
    x_corr = x - scatter_max(x, batch, dim=0)[0][batch]
    z = -(torch.rand_like(x)).log() / x_corr.exp()
    return scatter_min(z, batch, dim=dim)

def greedy_solve(env, tau=0, init_from_best=False, max_steps=None, max_time=None, verbose=False, ret_log=False, device='cpu'):
    if init_from_best:
        states = env.tradj.get_best_states(flat=True).numpy()
        scores, peeks = env.tradj._calc_scores_and_peeks(states)
        env.tradj._scores = scores
        env.tradj._node_features = env.tradj._init_node_features(states, peeks)
    peeks, actions = env.tradj.get_peeks().max(1)

    if max_time is None and max_steps is None:
        done = (peeks <=0).to(device)
    else:
        done = torch.tensor([False,]).to(device)

    t_tot = 0
    if max_steps is not None:
        if max_steps < 0:
            max_steps = int( abs(max_steps) * env.tradj.num_nodes_max )
    i = 0
    if ret_log:
        log = {
            "scores": [],
            "times": []
        }
    while not done.all():
        # peeks, actions = env.tradj.get_peeks().max(1)
        # peeks, actions = soft_greedy_action_select(env.tradj.get_peeks(), tau=tau)
        peeks = env.tradj.get_peeks(flat=True)
        t0 = time.time()
        peeks, actions = scatter_softmax_sample(peeks.to(device) / max(tau, 1e-9), env._batch_idx.to(device), 0)
        actions -= env._batch_ptr[:-1][:, None].to(actions)
        _, _ = env.step(actions, ret_reward=False)
        t_tot += time.time() - t0

        if ret_log:
            log["scores"].append(env.tradj.get_scores())
            log["times"].append(t_tot)

        i += 1

        if max_time is None and max_steps is None:
            peeks = env.tradj.get_peeks(flat=False)
            is_local_min = (peeks <= 0).all(1)
            done = done | is_local_min.to(done)
            if verbose:
                print(f"Step {i}: {done.sum()}/{done.numel()} environments finished.", end="\r")
        elif max_steps is not None:
            done = torch.tensor([i >= max_steps,])
            if done.all():
                print(f"Finishing. Max steps ({max_steps}) exceed.")
        else:
            done = torch.tensor([t_tot >= max_time,])
            if done.all():
                print(f"Max time of {max_time}s expired - finished after {i} steps.")

    if ret_log:
        return env, log
    else:
        return env

class DQNSolver(NetworkSolver):

    def __init__(self,
                 gnn,
                 rnn,
                 gnn2rnn,
                 node_encoder=None,
                 node_classifier=None,
                 add_glob_to_nodes=False,
                 add_glob_to_obs=False,
                 detach_gnn_from_rnn=False,
                 node_features=[NodeObservation.STATE, NodeObservation.PEEK_NORM],
                 global_features=[],

                 allow_reversible_actions=True,
                 intermediate_reward=0,
                 revisit_reward=0,

                 use_double_q_networks=True,
                 use_gnn_embeddings_from_buffer=False,
                 use_ecodqn=False,
                 default_actor_idx=None,
                 tau=0,
                 alpha=0,
                 munch_lower_log_clip=-1,

                 # Exploration
                 initial_epsilon=1,
                 final_epsilon=0.05,
                 final_epsilon_step=1000,

                 device=None):
        super().__init__(device=device, out_device=None)

        self._use_double_q_networks = use_double_q_networks
        self._use_gnn_embeddings_from_buffer = use_gnn_embeddings_from_buffer
        self._use_ecodqn = use_ecodqn
        if not self._use_ecodqn:
            actor = Actor_DoubleQ if self._use_double_q_networks else Actor_Q
        else:
            if self._use_double_q_networks:
                raise NotImplementedError
            else:
                actor = Actor_Q_ECODQN

        self.actor = actor(
            gnn=gnn,
            rnn=rnn,
            gnn2rnn=gnn2rnn,
            node_encoder=node_encoder,
            node_classifier = node_classifier,
            add_glob_to_nodes=add_glob_to_nodes,
            add_glob_to_obs=add_glob_to_obs,
            default_actor_idx=default_actor_idx,
            detach_gnn_from_rnn=detach_gnn_from_rnn,
            device=self.device
        )
        # self.tau = nn.Parameter(tau * torch.ones(1, device=self.device), requires_grad=True)
        # Parameters for soft and munchausen DQN.
        # tau: temperature for soft-DQN
        # alpha: scale for munchausen-DQN.  Note that this is an extension of soft-DQN.
        self.tau = tau
        self.alpha = alpha
        self.munch_lower_log_clip = munch_lower_log_clip
        if self.tau == 0 and self.alpha == 0:
            self._is_soft_dqn = False
        else:
            self._is_soft_dqn = True

        self.node_features = node_features
        self.global_features = global_features

        self.allow_reversible_actions = allow_reversible_actions
        self.intermediate_reward = intermediate_reward
        self.revisit_reward = revisit_reward

        # Prepare epochal logging.
        self.num_env_steps = 0
        self.num_rollouts = 0
        self.num_param_steps = 0
        self.initial_epsilon = initial_epsilon
        self.final_epsilon = final_epsilon
        self.final_epsilon_step = final_epsilon_step

    def parameters(self):
        return self.actor.parameters()

    def get_epsilon(self):
        return self.initial_epsilon - (self.initial_epsilon - self.final_epsilon) * min(1,
                                                                                        self.num_param_steps / self.final_epsilon_step)

    def train(self):
        self.actor.train()
        self.set_training(True)

    def test(self):
        self.actor.test()
        self.set_training(False)

    @torch.no_grad()
    def rollout(
            self,
            batch,
            num_tradj=1,
            num_steps=-1,
            log_stats=False,
            callbacks=None,
            use_epsilon=True,
            use_network_initialisation=False,
            tau=None,
            max_time=None,
            pre_solve_with_greedy=False,
            post_solve_with_greedy_from_best=False
    ):
        num_graphs = len(batch.info.num_nodes)
        log = defaultdict(list)

        _time_tot = 0

        if tau is None:
            tau = self.tau

        def log_step(env, actions, timings, scores0=None, peeks=None, state=None):
            """Log info about the tradjectory.

            Each log will have n entries for an n-step tradjectory.
            The i-th entry will contain:
                'scores_0': the score after i steps.
                'scores': the score after i+1 steps.
                'entropies': the entropies of the (i+1)-th action distributions.
                't_step': the time taken for the (i+1)-th inference + env step.
                't_inf': the time taken for the (i+1)-th inference.
                'probs': the action probabilties for the (i+1)-th step.
                'actions': the actions taken for the (i+1)-th step.
                'peeks': the peeks before taking the (i+1)-th step.
            """
            if scores0 is None:
                scores0 = log['scores'][-1].clone()

            log['scores'].append(env.tradj.get_scores().clone())
            log['scores0'].append(scores0)
            for k,v in timings.items():
                log[k].append(v)

            new_best = (log['scores'][-1] > log['scores0'][-1]) & (log['scores'][-1]==env.best_scores)
            log['t_opt'][new_best] = _time_tot

            if log_stats:
                assert peeks is not None, "must log peeks if log_stats is True"
                # probs: [num_graph, num_tradj, max_graph_size] --> [num_graph, max_graph_size, num_tradj]
                log['actions'].append(actions.detach().cpu())
                log['peeks'].append(peeks)
                log['state'].append(state)

        t0 = time.time()
        self.actor.initialise_embeddings(batch, num_tradj)
        _time_tot += (time.time() - t0)
        log['t_init_emb'].append(torch.tensor([time.time() - t0, ]))

        if use_network_initialisation:
            # Sample initial configuration from GNN embeddings.
            # Note that, if the actor does not have a head to predict a per-node
            # distribution, init_dist is returned as None, and hence we fall back
            # to the default (random) init of environment as if use_network_initialisation
            # was passed as False.
            init_dist = self.actor.get_node_dist_from_gnn_embeddings()
            if init_dist:
                init_state = init_dist.sample()
                if init_state.size(-1)==1:
                    init_state = init_state.squeeze(-1)
                else:
                    # Randomly pick which readout head we are using.
                    init_state = init_state[..., torch.randint(0, init_state.size(-1), (1,))].squeeze(-1)
                    # Apply random perturbations with probability epsilon.
                    if use_epsilon:
                        replace_mask = torch.rand_like(init_state) < self.get_epsilon()
                        init_state[replace_mask] = torch.randint_like(init_state[replace_mask], 0, 2)
            else:
                init_state = None

        else:
            # Use default (random) init of environment.
            init_state = None

        t0 = time.time()
        env = Environment(
            batch.nx,
            num_tradj,
            batch.tg,
            node_features=self.node_features,
            glob_features=self.global_features,
            init_state=init_state,
            intermediate_reward=self.intermediate_reward,
            revisit_reward=self.revisit_reward,
            add_edges_to_observations=self._use_ecodqn,
            device=self.device
        )
        if pre_solve_with_greedy:
            env = greedy_solve(env, init_from_best=False, verbose=False)

        log['t_setup'].append(torch.tensor([time.time()-t0,]))

        init_score = env.tradj.get_scores()
        log['init_score'] = init_score

        # mask, update_mask_fn = prepare_mask(batch)
        done = torch.zeros(num_graphs, num_tradj, device=self.device)
        log['t_opt'] = torch.ones(num_graphs, num_tradj) * _time_tot

        if num_steps < 0:
            num_steps = np.abs(num_steps) * batch.info.num_nodes
        else:
            num_steps = torch.tensor([num_steps] * num_graphs)

        # observation at t=0.
        obs = env.observe()
        scores_init = env.scores.clone()

        t_start = time.time()
        for t in range(num_steps.max()):

            if (max_time is not None) and _time_tot >= max_time:
                print(f"Max time exceeded, terminating optimisation after {t} steps.")
                break

            if log_stats:
                peeks = env.tradj.get_peeks().detach().cpu().clone()
                state = env.tradj.get_states().detach().cpu().clone()
            else:
                peeks = None
                state = None

            if self.is_training:
                actor_state = self.actor.get_state()

            t0 = time.time()
            actions, q_vals, timings = self.actor.action_select(obs,
                                                       epsilon=self.get_epsilon() if use_epsilon else 0,
                                                       tau=tau,
                                                       return_q_values=True,
                                                       return_timings=True)
            timings['t_inf'] = time.time() - t0

            obs_new, reward = env.step(actions, ret_reward=self.is_training)
            timings['t_step'] = time.time() - t0
            _time_tot += timings['t_step']

            if self.is_training:
                self.num_env_steps += 1
                done[t + 1 == num_steps, :] = 1

            if callbacks is not None and self.is_training:
                for (fn, env_step_freq, param_step_freq) in callbacks:
                    if env_step_freq is not None and self.num_env_steps % env_step_freq == 0:
                        # actor_state, obs, action, reward, done
                        fn(self, actor_state=actor_state, obs=obs, action=actions, reward=reward,
                           done=done, t_step=t)
                    if param_step_freq is not None and self.num_param_steps % param_step_freq == 0:
                        fn(self, actor_state=actor_state, obs=obs, action=actions, reward=reward,
                           done=done, t_step=t)

            obs = obs_new
            del obs_new

            log_step(env, actions, timings, scores_init, peeks, state)
            scores_init = None

        if self.is_training:
            self.num_rollouts += num_graphs * num_tradj

        if post_solve_with_greedy_from_best:
            env = greedy_solve(env, init_from_best=True, verbose=False)
            log_step(env,
                     -torch.ones_like(actions),
                     timings={}, scores0=None, peeks=None, state=None)

        log['best_state'] = env.tradj.get_best_states(flat=False)
        log['t_rollout'] = torch.tensor([time.time() - t_start,])
        log['t_tot'] = torch.tensor([_time_tot, ])

        return log

    def accumulate_gradients(self, sample, gamma=1, init_bce_weight=1):
        assert self.is_training, "Must be training (.is_training) to accumulate gradients."
        return self._accumulate_gradients_soft_dqn(sample, gamma, init_bce_weight)

    def _accumulate_gradients_soft_dqn(self, sample, gamma=1, init_bce_weight=1):

        t_eval = sample.k_BPTT.to(self.device)
        t_bootstrap = (sample.k_BPTT + sample.n_step).to(self.device)

        if self._use_double_q_networks:
            q_values = [torch.zeros(len(t_eval), device=self.device) for _ in range(2)]
            values_next = [torch.zeros(len(t_eval), device=self.device) for _ in range(2)]
            muchausen_bootstraps = [torch.zeros(len(t_eval), device=self.device) for _ in range(2)]
        else:
            q_values = torch.zeros(len(t_eval), device=self.device)
            values_next = torch.zeros(len(t_eval), device=self.device)
            muchausen_bootstraps = torch.zeros(len(t_eval), device=self.device)

        ###
        # Main network loop.
        ###

        # prepare actor to correct initial state:
        #  initialise_embeddings --> generate per-node embeddings.
        self.actor.initialise_node_embeddings(sample.batch, num_tradj=1, use_target_network=False)

        # Calculate loss for the node initialisation if configured.
        dist = self.actor.get_node_dist_from_gnn_embeddings()
        if dist and (init_bce_weight != 0):
            # dist.probs : [num_nodes, num_tradj=1, num_heads]
            # sample.batch.tg.best_state : [num_nodes, 1]
            probs = dist.probs.squeeze(1)
            target = sample.batch.tg.best_state.expand_as(probs).to(self.device)
            loss_bce = F.binary_cross_entropy(probs, target, reduction='none')
            # loss_bce: [num_nodes, num_heads]
            loss_bce = scatter_mean(loss_bce, sample.batch.tg.batch.to(self.device), dim=0)
            loss_bce = loss_bce.min(-1).values.mean()
        else:
            # Either we don't have network initialisation configured, or are simply
            # not training on it.
            loss_bce = 0

        # set_state --> set internal state (RNN hidden state + any memory).
        self.actor.set_state(
            sample.actor_state_start,
            use_target_network=False,
            overwite_init_node_embeddings=self._use_gnn_embeddings_from_buffer,
            reset_internal_state=False,
        )

        obs_list, actions = sample.obs, sample.action.to(self.device)

        for t, (obs, act) in enumerate(zip(obs_list, actions)):

            select_q_val = (t_eval == t)

            # Note that the internal state of the RNN is updated using the last selected
            # node embeddings and obs internally in get_Q_values().
            q_vals = self.actor.get_Q_values(
                obs,
                use_target_network=False,
                mask=None,
                actor_idx=None,
            )

            if select_q_val.any():
                if self._use_double_q_networks:
                    q_values[0][select_q_val] = q_vals[0].squeeze()[act.squeeze()][select_q_val]
                    q_values[1][select_q_val] = q_vals[1].squeeze()[act.squeeze()][select_q_val]
                else:
                    q_values[select_q_val] = q_vals.squeeze()[act.squeeze()][select_q_val]

            # Prime the policy and q networks with the next set of action/node embeddings.
            self.actor.get_and_cache_action_embeddings(obs, act, stop_grad=True)

        ###
        # Target network loop.
        ###'''

        def to_scaled_log_probs(q_vals, tau, batch):
            """log-sum-exp trick for computing tau * log_probs.

            See appendix B.1 of the M-DQN paper (2007.14430).
            """
            v_vals = scatter_max(q_vals, batch, dim=0)[0]
            adv_vals = q_vals - v_vals[batch]
            return adv_vals - tau * scatter_sum( (adv_vals / tau).exp(), batch, dim=0).log()[batch]

        def to_munchausen_bootstrap(q_vals, action, lower_bound=None):
            if q_vals.dim() > 1:
                q_vals.squeeze_(1)
            if action.dim() > 1:
                action = action.squeeze() # not in place to keep dimensions correct for later..
            tau = max(self.tau, 1e-3)
            batch = obs.batch_idx.to(q_vals.device)
            # action_probs = to_probs(q_vals, tau, batch)[action]
            # value = (tau * action_probs.log())
            value = to_scaled_log_probs(q_vals, tau, batch)[action]
            if lower_bound is not None:
                value.clamp_(min=lower_bound)
            return self.alpha * value

        def to_values(q_vals, obs):
            '''
            Coverts q_values to the value of the state, according to
            a soft-DQN estimation with temperature tau.

            The value is:
                scatter_sum(probs * (q_vals - tau * probs.log()), batch, dim=0)

            However,using the log-sum-exp trick we find the scaling factor,
                (q_vals - tau * probs.log()),
            is independent of the chosen action as the q_vals cancel out.  Hence we can
            rewrite this as:
                scatter_sum(probs, batch, dim=0) * values
            where the sum of probs is clearly constant, and:
                values = v_val + tau * log_sum_exp( (q_vals - v_val)/tau ),
            with v_val the maximum q_value.

            This seems to follow from appendix B.1 of the M-DQN paper (2007.14430), but is
            never explicitly stated.
            '''
            if q_vals.dim() > 1:
                q_vals.squeeze_(1)
            tau = max(self.tau, 1e-9)
            batch = obs.batch_idx.to(q_vals.device)
            v_vals = scatter_max(q_vals, batch, dim=0)[0]
            values = v_vals + \
                           tau * scatter_sum(((q_vals - v_vals[batch]) / tau).exp(), batch, dim=0).log()
            return values

        # prepare actor to correct initial state:
        # 1. initialise_embeddings --> generate per-node embeddings.
        # 2. set_state --> set internal state (RNN hidden state + any memory).
        with torch.no_grad():
            if not self._use_gnn_embeddings_from_buffer:
                self.actor.initialise_node_embeddings(sample.batch, num_tradj=1, use_target_network=True)
            self.actor.set_state(
                sample.actor_state_start,
                use_target_network=True,
                overwite_init_node_embeddings=self._use_gnn_embeddings_from_buffer,
                reset_internal_state = False,
            )

            for t, (obs, act) in enumerate(zip(obs_list, actions)):

                select_q_val = (t_eval == t)
                select_next_q_val = (t_bootstrap == t)

                q_vals = self.actor.get_Q_values(obs, use_target_network=True, actor_idx=None, mask=None)

                if select_q_val.any() and self.alpha != 0:
                    # We've already got q_values from the forwards pass, but will now add the Muchausen
                    # RL bootstrap if required (i.e. if alpha != 0).
                    if self._use_double_q_networks:
                        # Bootstrap each network with agents target policy.
                        if self.actor.default_actor_idx is not None:
                            q = q_vals[self.actor.default_actor_idx]
                        else:
                            q = torch.stack(q_vals, dim=-1).min(-1).values
                        bootstrap = to_munchausen_bootstrap(
                            q, act, lower_bound=self.munch_lower_log_clip
                        )
                        muchausen_bootstraps[0][select_q_val] = bootstrap[select_q_val]
                        muchausen_bootstraps[1][select_q_val] = bootstrap[select_q_val]

                    else:
                        muchausen_bootstraps[select_q_val] = to_munchausen_bootstrap(
                            q_vals, act, lower_bound=self.munch_lower_log_clip
                        )[select_q_val]

                if select_next_q_val.any():
                    if self._use_double_q_networks:
                        values_next[0][select_next_q_val] = to_values(q_vals[0], obs)[select_next_q_val]
                        values_next[1][select_next_q_val] = to_values(q_vals[1], obs)[select_next_q_val]
                    else:
                        values_next[select_next_q_val] = to_values(q_vals, obs)[select_next_q_val]

                self.actor.get_and_cache_action_embeddings(obs, act, stop_grad=True)

        reward = sample.reward.to(self.device)
        done = sample.done.to(self.device)

        if self._use_double_q_networks:
            values_next = torch.stack(values_next, dim=-1).min(-1).values
        td_target = reward + (1 - done) * gamma ** sample.n_step.to(self.device) * values_next

        if self._use_double_q_networks:
            loss = F.smooth_l1_loss(
                q_values[0], td_target + muchausen_bootstraps[0], beta=1
            ) + F.smooth_l1_loss(
                q_values[1], td_target + muchausen_bootstraps[1], beta=1
            )
        else:
            loss = F.smooth_l1_loss(q_values, td_target + muchausen_bootstraps, beta=1)

        loss = loss + init_bce_weight*loss_bce

        loss.backward()

        return loss.detach().cpu().item()


    def save(self, fname, quiet=False):
        dir = os.path.dirname(fname)
        if dir != '':
            mk_dir(dir, quiet=True)

        checkpoint = {
            'actor:checkpoint': self.actor.get_checkpoint(),
            'num_env_steps': self.num_env_steps,
            'num_rollouts': self.num_rollouts,
            'num_param_steps': self.num_param_steps,
            'initial_epsilon': self.initial_epsilon,
            'final_epsilon' : self.final_epsilon,
            'final_epsilon_step' : self.final_epsilon_step,
        }
        if os.path.splitext(fname)[-1] != '.pth':
            fname += '.pth'
        torch.save(checkpoint, fname)
        if not quiet:
            print("Saved DQNSolver model to {}.".format(fname))
        return fname

    def load(self, fname, quiet=False):
        checkpoint = torch.load(fname)

        self.actor.load_checkpoint(checkpoint['actor:checkpoint'])
        self.actor.set_device(self.device)

        self.num_env_steps = checkpoint['num_env_steps']
        self.num_rollouts = checkpoint['num_rollouts']
        self.num_param_steps = checkpoint['num_param_steps']

        self.initial_epsilon = checkpoint['initial_epsilon']
        self.final_epsilon = checkpoint['final_epsilon']
        self.final_epsilon_step = checkpoint['final_epsilon_step']

        if not quiet:
            print("Loaded DQNSolver from {}.".format(fname))