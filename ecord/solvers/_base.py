from abc import ABC, abstractmethod
from collections import namedtuple, defaultdict

import torch

Rollout = namedtuple('Rollout', ['best_score', 'best_state', 'log'])


class Learner:

    def __init__(self,
                 trainable=False,
                 is_training=False):
        self.trainable = trainable
        self.is_training = is_training and trainable

    def store_transition(self, transition):
        pass

    def learning_step(self, accumulate_gradients=False):
        if not self.trainable or not self.is_training:
            pass
        else:
            raise NotImplementedError()

    def set_training(self, is_training):
        if self.trainable:
            self.is_training = is_training


class Solver(Learner, ABC):

    def __init__(self,
                 trainable=False,
                 is_training=False):
        super().__init__(trainable, is_training)
        self.reset_log()

    def observe(self, action, obs, rewards):
        self.log['actions'].append(action)
        self.log['obs'].append(obs)
        self.log['rewards'].append(rewards)

    def reset_log(self):
        self.log = defaultdict(list)

    @abstractmethod
    def rollout(self, env, step_factor=1):
        raise NotImplementedError

    @abstractmethod
    def save(self, fname, quiet=False):
        raise NotImplementedError

    @abstractmethod
    def load(self, fname, quiet=False):
        raise NotImplementedError


class NetworkSolver(Solver):

    def __init__(self,
                 device=None,
                 out_device="cpu"):
        super().__init__(trainable=True, is_training=True)

        if device is None:
            if torch.cuda.is_available():
                self.device = 'cuda'
                print("GPU found!", end=" ")
            else:
                self.device = 'cpu'
                print("GPU not found!", end=" ")
        else:
            # I trust you know what you are doing!
            self.device = device
        if out_device is None:
            self.out_device = self.device
        else:
            self.out_device = out_device
        print(f"Creating solver on device '{self.device}' with output on '{self.out_device}'.")

    def train(self):
        self.set_training(True)

    def test(self):
        self.set_training(False)