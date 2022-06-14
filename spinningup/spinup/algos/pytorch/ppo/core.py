import numpy as np
import scipy.signal
from gym.spaces import Box, Discrete

import torch
import torch.nn as nn
from torch.distributions.normal import Normal
from torch.distributions.categorical import Categorical


def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)


def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes) - 1):
        act = activation if j < len(sizes) - 2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j + 1]), act()]
    return nn.Sequential(*layers)


def conv(conv_sizes=(32, 64, 64), dense_sizes=(512, 128), activation=nn.ReLU()):
    return nn.Sequential(
        nn.Conv2d(5, conv_sizes[0], 3, 1),
        activation,
        nn.Conv2d(conv_sizes[0], conv_sizes[1], 3, 1),
        activation,
        nn.Conv2d(conv_sizes[1], conv_sizes[2], 3, 1),
        activation,
        nn.Flatten(),
        nn.Linear(18*18*conv_sizes[-1], dense_sizes[0]),
        activation,
        nn.Linear(dense_sizes[0], dense_sizes[1]),
        activation
    )


def conv_last(out_size=2, input_size=130):
    return nn.Sequential(
        nn.Linear(input_size, out_size),
        nn.Tanh()
    )


def count_vars(module):
    return sum([np.prod(p.shape) for p in module.parameters()])


def discount_cumsum(x, discount):
    """
    magic from rllab for computing discounted cumulative sums of vectors.

    input: 
        vector x, 
        [x0, 
         x1, 
         x2]

    output:
        [x0 + discount * x1 + discount^2 * x2,  
         x1 + discount * x2,
         x2]
    """
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]


class Actor(nn.Module):

    def _distribution(self, field, agent_center):
        raise NotImplementedError

    def _log_prob_from_distribution(self, pi, act):
        raise NotImplementedError

    def forward(self, obs, act=None):
        # Produce action distributions for given observations, and 
        # optionally compute the log likelihood of given actions under
        # those distributions.
        field = torch.as_tensor(obs['world'], dtype=torch.float32)
        position = torch.as_tensor(obs['agent_center'])

        pi = self._distribution(field, position)
        logp_a = None
        if act is not None:
            logp_a = self._log_prob_from_distribution(pi, act)
        return pi, logp_a

class MLPGaussianActor(Actor):

    def __init__(self, act_dim):
        super().__init__()
        log_std = -0.5 * np.ones(act_dim, dtype=np.float32)
        self.log_std = torch.nn.Parameter(torch.as_tensor(log_std))
        self.mu_net = conv()
        self.mu_net_last = conv_last()

    def _distribution(self, field, agent_center):
        obs = torch.as_tensor(field)
        if len(obs.shape) == 3:
            obs = obs.unsqueeze(0)

        emb = self.mu_net(obs)
        if len(agent_center.shape) == 1:
            agent_center = agent_center.unsqueeze(0)
        emb = torch.concat([emb, agent_center], dim=1)
        mu = self.mu_net_last(emb)

        if mu.shape[0] == 1:
            mu = mu.squeeze(0)

        std = torch.exp(self.log_std)
        return Normal(mu, std)

    def _log_prob_from_distribution(self, pi, act):
        return pi.log_prob(act).sum(axis=-1)  # Last axis sum needed for Torch Normal distribution


class MLPCritic(nn.Module):

    def __init__(self):
        super().__init__()
        self.v_net = conv()
        self.v_net_end = conv_last(out_size=1)

    def forward(self, world, agent_center):
        world = torch.as_tensor(world)
        if len(world.shape) == 3:
            world = world.unsqueeze(0)

        emb = self.v_net(world)

        if len(agent_center.shape) == 1:
            agent_center = agent_center.unsqueeze(0)

        emb = torch.concat([emb, agent_center], dim=1)
        out = self.v_net_end(emb)

        if out.shape[0] == 1:
            out = out.squeeze(0)

        return torch.squeeze(out, -1)  # Critical to ensure v has right shape.


class MLPActorCritic(nn.Module):

    def __init__(self, observation_space, action_space,
                 hidden_sizes=(64, 64), activation=nn.Tanh):
        super().__init__()

        obs_dim = np.prod(observation_space.shape)

        # policy builder depends on action space
        if isinstance(action_space, Box):
            self.pi = MLPGaussianActor(action_space.shape[0])
        elif isinstance(action_space, Discrete):
            raise Exception()

        # build value function
        self.v = MLPCritic()

    def step(self, obs):
        field = torch.as_tensor(obs['world'], dtype=torch.float32)
        position = torch.as_tensor(obs['agent_center'], dtype=torch.float32)/1536

        with torch.no_grad():
            pi = self.pi._distribution(field, position)
            a = pi.sample()
            logp_a = self.pi._log_prob_from_distribution(pi, a)
            v = self.v(field, position)
        return a.numpy(), v.numpy(), logp_a.numpy()

    def act(self, obs):
        return self.step(obs)[0]
