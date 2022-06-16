import numpy as np
import scipy.signal
from gym.spaces import Box, Discrete

import torch
import torch.nn as nn
from torch.distributions.normal import Normal

from spinup.algos.pytorch.helpers.general_nets import conv, conv_last, CustomAct


def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)

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


class MLPGaussianActor(Actor):
    def __init__(self, act_dim, conv_sizes, dense_sizes, activation_conv, device='cpu'):
        super().__init__()
        self.device = device

        log_std = -0.5 * np.ones(act_dim, dtype=np.float32)
        self.log_std = torch.nn.Parameter(torch.as_tensor(log_std).to(device))
        self.mu_net = conv(conv_sizes=conv_sizes, dense_sizes=dense_sizes, activation=activation_conv)
        self.mu_net_last = conv_last(out_size=2, input_size=dense_sizes[-1]+2, activation=CustomAct)

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

    def forward(self, obs, act=None):
        field = torch.as_tensor(obs['world'], dtype=torch.float32).to(self.device)
        position = torch.as_tensor(obs['agent_center']).to(self.device)

        pi = self._distribution(field, position)
        logp_a = None
        if act is not None:
            logp_a = self._log_prob_from_distribution(pi, act)
        return pi, logp_a


class MLPCritic(nn.Module):
    def __init__(self, conv_sizes, dense_sizes, activation_conv, device='cpu'):
        super().__init__()
        self.device = device

        self.v_net = conv(conv_sizes=conv_sizes, dense_sizes=dense_sizes, activation=activation_conv)
        self.v_net_end = conv_last(out_size=1, input_size=dense_sizes[-1]+2, activation=nn.Tanh)

    def forward(self, world, agent_center):
        world = torch.as_tensor(world).to(self.device)
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

    def __init__(self, action_space, device, conv_sizes=(64, 32, 16), dense_sizes=(512, 128), activation_conv=nn.ReLU):
        super().__init__()

        self.device = device

        # policy builder depends on action space
        if isinstance(action_space, Box):
            self.pi = MLPGaussianActor(action_space.shape[0], conv_sizes=conv_sizes, dense_sizes=dense_sizes, activation_conv=activation_conv, device=device).to(device)
        elif isinstance(action_space, Discrete):
            raise Exception()

        # build value function
        self.v = MLPCritic(conv_sizes=conv_sizes, dense_sizes=dense_sizes, activation_conv=activation_conv, device=device).to(device)

    def step(self, obs):
        field = torch.as_tensor(obs['world'], dtype=torch.float32).to(self.device)
        position = torch.as_tensor(obs['agent_center'], dtype=torch.float32).to(self.device)/1536

        with torch.no_grad():
            pi = self.pi._distribution(field, position)
            a = pi.sample()
            logp_a = self.pi._log_prob_from_distribution(pi, a)
            v = self.v(field, position)
        return a.numpy(), v.numpy(), logp_a.numpy()

    def act(self, obs):
        return self.step(obs)[0]
