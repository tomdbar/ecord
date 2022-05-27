import os
import random
import time
from collections import defaultdict

import numpy as np
import torch

from ecord.validate.testing import test_solver
from ecord.utils import mk_dir
from ecord.train.buffer import ReplayBuffer


class TrainingFinished(Exception):
    pass


class Trainer:

    def __init__(self,
                 solver,
                 graph_batcher,

                 training_gg,
                 test_graph_configs = [],

                 batch_size=32,
                 num_sub_batches=1,
                 lr=5e-4,
                 env_steps_per_update=1,
                 update_target_frequency=100,
                 update_target_polyak=1,

                 # Buffer
                 buffer_capacity=1000,
                 buffer_add_probability=1,
                 k_BPTT=5,
                 n_step=2,
                 min_buffer_length=100,
                 crop_tradjectories_at_final_reward = False,
                 add_final_scores_to_transitions = False,

                 # Training
                 gamma=0.99,
                 init_bce_weight = 0,
                 prob_network_initialisation = 0,

                 num_parallel_graphs=16,
                 tradj_per_graph=1,
                 num_steps_per_rollout=-2,

                 save_loc='./',
                 checkpoint_freq=None,
                 ):

        self.solver = solver
        self.graph_batcher = graph_batcher
        self.gg = training_gg

        self.test_graph_configs = test_graph_configs

        self.num_parallel_graphs = num_parallel_graphs
        self.tradj_per_graph = tradj_per_graph
        self.num_steps_per_rollout = num_steps_per_rollout

        self.batch_size = batch_size
        self.num_sub_batches = num_sub_batches
        self._effective_batch_size = batch_size // num_sub_batches
        self.env_steps_per_update = env_steps_per_update
        self.crop_tradjectories_at_final_reward = crop_tradjectories_at_final_reward
        self.add_final_scores_to_transitions = add_final_scores_to_transitions

        self.update_target_frequency = update_target_frequency
        self.update_target_polyak = update_target_polyak

        self.gamma = gamma
        self.init_bce_weight = init_bce_weight
        self.prob_network_initialisation = prob_network_initialisation

        # prepare buffer
        self.k_BPTT, self.n_step = k_BPTT, n_step
        self.buffer = ReplayBuffer(
            self.graph_batcher,
            buffer_capacity,
            add_probability=buffer_add_probability,
            k_BPTT=k_BPTT,
            n_step=n_step,
            gamma=gamma,
        )
        self.min_buffer_length = min_buffer_length

        self.lr = lr
        self.optimizer = self.reset_optimizer()

        self.log = defaultdict(list)

        self.save_loc = save_loc
        self._best_test_scores = [None] * len(self.test_graph_configs)
        self._best_test_mean_scores = [None] * len(self.test_graph_configs)
        self.checkpoint_freq = checkpoint_freq

    def reset_optimizer(self):
        self.optimizer = torch.optim.Adam(self.solver.parameters(), lr=self.lr)
        return self.optimizer

    def train(self, num_steps=100, test_freq=10, verbose=True, save_final=True):

        num_steps_start = self.solver.num_param_steps
        num_steps_finish = num_steps_start + num_steps

        def _on_batch_start():
            if self.crop_tradjectories_at_final_reward:
                self._t_final_reward = -1*np.ones((self.num_parallel_graphs, self.tradj_per_graph))
            else:
                self._t_final_reward = None

        def _on_batch_end(best_states=None):
            if self.crop_tradjectories_at_final_reward or self.add_final_scores_to_transitions:
                self.buffer.flush(self._t_final_reward, best_states)

        def update_buffer(solver, actor_state, obs, action, reward, done, t_step):
            self.buffer.append(
                actor_state = actor_state,
                obs = obs,
                action = action,
                reward = reward,
                done = done,
                flush = not (self.crop_tradjectories_at_final_reward or self.add_final_scores_to_transitions)
            )
            if self.crop_tradjectories_at_final_reward:
                self._t_final_reward[reward != 0] = t_step

        def learning_step(solver, *args, **kwargs):
            if len(self.buffer) > self.min_buffer_length:
                actor_state = solver.actor.get_state()

                t = time.time()
                loss = 0
                for i_batch in range(self.num_sub_batches):
                    sample = self.buffer.sample(self._effective_batch_size)
                    with torch.enable_grad():
                        loss += solver.accumulate_gradients(
                            sample,
                            gamma=self.gamma,
                            init_bce_weight=self.init_bce_weight,
                        ) / self.num_sub_batches

                self.optimizer.step()
                self.optimizer.zero_grad()
                
                solver.num_param_steps += 1

                self.log['loss'].append((solver.num_param_steps, loss))
                self.log['step_time'].append((solver.num_param_steps, time.time()-t))

                if solver.num_param_steps % self.update_target_frequency == 0:
                    solver.actor.update_target_network(self.update_target_polyak)

                solver.actor.set_state(actor_state, overwite_init_node_embeddings=True)

                if solver.num_param_steps >= num_steps_finish:
                    raise TrainingFinished()

        def save_checkpoint(solver, *args, **kwargs):
            if solver.num_param_steps > 0:
                self.save(f"checkpoints/{solver.num_param_steps}steps.pth", quiet=False)

        @torch.no_grad()
        def test(solver, *args, **kwargs):
            if solver.num_env_steps == 1 or len(self.buffer) > self.min_buffer_length:
                # Run periodic tests.
                actor_state = solver.actor.get_state()
                was_training = solver.is_training
                solver.test()

                test_res = []
                for idx, config in enumerate(self.test_graph_configs):

                    log = test_solver(
                        solver=solver,
                        graph_batcher=self.graph_batcher,
                        config=config,
                        log_stats=False,
                        verbose=False,
                    )

                    scores_max, scores_mean = log['apx_max'], log['apx_mean']
                    if scores_max is None:
                        scores_max, scores_mean = log['scores_max'], log['scores_mean']
                    scores_mean_idx = log['scores_idx_mean']
                    init_score_mean = log['init_apx_score_mean']
                    if init_score_mean is None:
                        init_score_mean = log['init_score_mean']

                    test_score, test_score_mean, test_score_idx_mean, test_init_score_mean = (
                        scores_max.mean(), scores_mean.mean(), scores_mean_idx.mean(), init_score_mean.mean())
                    test_res.append((test_score, test_score_mean, test_score_idx_mean, test_init_score_mean))


                    if (
                            (self._best_test_scores[idx] is None)
                            or (test_score > self._best_test_scores[idx])
                            or ((test_score == self._best_test_scores[idx]) and test_score_mean > self._best_test_mean_scores[idx])
                    ):
                        # Save the current best solver by test score.
                        self._best_test_scores[idx] = test_score
                        self.solver.save(
                            self.__format_checkpoint_fname(f"solver_best_{config.get_label()}.pth"),
                            quiet=True
                        )

                    if (
                            (self._best_test_mean_scores[idx] is None)
                            or (test_score_mean > self._best_test_mean_scores[idx])
                    ):
                        # Save the current best solver by test score.
                        self._best_test_mean_scores[idx] = test_score_mean
                        self.solver.save(
                            self.__format_checkpoint_fname(f"solver_best_mean_{config.get_label()}.pth"),
                            quiet=True
                        )

                self.log['scores'].append((solver.num_param_steps, test_res))

                if was_training:
                    solver.train()
                solver.actor.set_state(actor_state, overwite_init_node_embeddings=True)

                # Print test output and other statistics.
                if verbose:

                    if len(self.log['loss']) < test_freq:
                        loss_str = "N/A"
                    else:
                        loss_str = f"{np.mean([l for n,l in self.log['loss'][-test_freq:]]):.3E}"

                    log_str = f"Steps: {solver.num_param_steps}" + \
                              f" | loss : {loss_str}"
                    if self.init_bce_weight == 0:
                        log_str += " | score (mean/step) : "
                    else:
                        log_str += " | score (init/mean/step) : "
                    _, test_res = self.log['scores'][-1]
                    for idx, (config, (scores_max, scores_mean, scores_idx, init_score_mean)) in enumerate(zip(self.test_graph_configs, test_res)):
                        if self.init_bce_weight == 0:
                            log_str += f"{config.get_label()} = {scores_max.item():.3f} ({scores_mean:.3f}/{scores_idx:.1f})"
                        else:
                            log_str += f"{config.get_label()} = {scores_max.item():.3f} ({init_score_mean:.3f}/{scores_mean:.3f}/{scores_idx:.1f})"

                        if idx < len(test_res) - 1:
                            log_str += ", "
                    if len(self.log['step_time'])>0:
                        t_step_str = f"{self.log['step_time'][-1][-1]*10**3:.3f}ms"
                    else:
                        t_step_str = "N/A"
                    log_str += f" | step time : {t_step_str}"
                    if len(self.log['t_rollout']) > 0:
                        t_roll_str = f"{self.log['t_rollout'][-1][0]*10**3:.3f}ms"
                    else:
                        t_roll_str = "N/A"
                    log_str += f", eps. time : {t_roll_str}."

                    print(log_str)

        # Callback : [function, freq/env_steps, freq/param_steps]
        callbacks = [
            (update_buffer, 1, None),
            (learning_step, self.env_steps_per_update, None),
            (save_checkpoint, None, self.checkpoint_freq),
            (test, None, test_freq),
        ]

        self.solver.train()
        while True:
            try:
                _on_batch_start()
                batch = self.graph_batcher([G for G in self.gg.generate(self.num_parallel_graphs)])
                self.buffer.prepare_for_batch(batch)
                log = self.solver.rollout(
                    batch,
                    num_tradj=self.tradj_per_graph,
                    num_steps=self.num_steps_per_rollout,
                    log_stats=False,

                    use_epsilon=True, # epsilon greedy exploration
                    use_network_initialisation = random.random() < self.prob_network_initialisation,

                    callbacks=callbacks
                )
                self.log['t_rollout'].append((np.sum(log['t_step']), np.mean(log['t_step'])))
                _on_batch_end(log['best_state'])
            except TrainingFinished:
                _on_batch_end(log['best_state'])
                break

        if save_final:
            self.save("checkpoints/final.pth", quiet=not verbose)

    def __format_checkpoint_fname(self, fname):
        '''Formats a checkpoint file location as an absolute path with a '.pth' file extension.
        If a relative path is passed, this returns the absolute path relative to self.save_loc.
        '''
        if os.path.splitext(fname)[-1] != '.pth':
            fname += '.pth'
        if not os.path.isabs(fname):
            fname = os.path.join(self.save_loc, fname)
        return fname

    def save(self, fname="trainer", save_buffer=True, quiet=False):
        fname = self.__format_checkpoint_fname(fname)
        if not quiet:
            print(f"Saving checkpoint {fname}.", end="...")

        dir = os.path.dirname(fname)
        if dir != '':
            mk_dir(dir, quiet=True)

        solver_fname = os.path.splitext(fname)[0] + '_solver'
        solver_fname = self.solver.save(solver_fname, quiet=True)

        if save_buffer:
            bff_fname = os.path.join(dir, os.path.splitext(fname)[-1] + "_buffer.dmp")
            checkpoint_bff = {'buffer':self.buffer.buffer}
            torch.save(checkpoint_bff, bff_fname)
        else:
            bff_fname = None

        checkpoint = {
            'solver:fname': solver_fname,
            'optimizer:state_dict': self.optimizer.state_dict(),
            'log': self.log,
            'best_test_scores': self._best_test_scores,
            'buffer_path': bff_fname,
        }

        torch.save(checkpoint, fname)
        if not quiet:
            print("...done.")

    def load(self, fname=None, quiet=False):
        if fname is None:
            fname = "checkpoints/final.pth"
        fname = self.__format_checkpoint_fname(fname)
        print("Loading checkpoint {}.".format(fname), end="...")
        try:
            checkpoint = torch.load(fname)
        except:
            checkpoint = torch.jit.load(fname)

        try:
            self.solver.load(checkpoint['solver:fname'], quiet=True)
        except:
            print(f"\tsolver not found (expected at {checkpoint['solver:fname']})")

        try:
            self.optimizer.load_state_dict(checkpoint['optimizer:state_dict'])
        except:
            print("\tOptimizer could not be loaded.")

        if checkpoint['buffer_path'] is not None:
            checkpoint_bff = torch.load(checkpoint['buffer_path'])
            self.buffer.buffer = checkpoint_bff['buffer']

        self.log = checkpoint['log']
        self._best_test_scores = checkpoint['best_test_scores']

        if not quiet:
            print("...done.")