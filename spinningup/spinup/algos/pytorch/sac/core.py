import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal

from spinup.algos.pytorch.helpers.general_nets import conv, conv_last, init_weights


def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)

def count_vars(module):
    return sum([np.prod(p.shape) for p in module.parameters()])


LOG_STD_MAX = 2
LOG_STD_MIN = -20


def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers)


class SquashedGaussianMLPActor(nn.Module):

    def __init__(self, act_limit, conv_sizes, dense_sizes, activation_conv, device=torch.device('cpu')):
        super().__init__()
        self.device = device

        self.net = conv(conv_sizes=conv_sizes, dense_sizes=dense_sizes, activation=activation_conv)

        self.mu_layer = conv_last(out_size=2, input_size=dense_sizes[-1]+2)
        self.log_std_layer = conv_last(out_size=2, input_size=dense_sizes[-1]+2, activation=nn.Tanh)

        self.act_limit = act_limit

    def forward(self, field, agent_center, deterministic=False, with_logprob=True):
        obs = torch.as_tensor(field).to(self.device)
        if len(obs.shape) == 3:
            obs = obs.unsqueeze(0)

        emb = self.net(obs)

        if len(agent_center.shape) == 1:
            agent_center = agent_center.unsqueeze(0)

        emb = torch.concat([emb, agent_center], dim=1)

        mu = self.mu_layer(emb)
        log_std = self.log_std_layer(emb)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(log_std)

        # Pre-squash distribution and sample
        pi_distribution = Normal(mu, std)
        if deterministic:
            # Only used for evaluating policy at test time.
            pi_action = mu
        else:
            pi_action = pi_distribution.rsample()

        if with_logprob:
            # Compute logprob from Gaussian, and then apply correction for Tanh squashing.
            # NOTE: The correction formula is a little bit magic. To get an understanding 
            # of where it comes from, check out the original SAC paper (arXiv 1801.01290) 
            # and look in appendix C. This is a more numerically-stable equivalent to Eq 21.
            # Try deriving it yourself as a (very difficult) exercise. :)
            logp_pi = pi_distribution.log_prob(pi_action).sum(axis=-1)
            logp_pi -= (2*(np.log(2) - pi_action - F.softplus(-2*pi_action))).sum(axis=1)
        else:
            logp_pi = None

        pi_action = torch.tanh(pi_action)

        return pi_action, logp_pi


class MLPQFunction(nn.Module):

    def __init__(self, conv_sizes, dense_sizes, activation_conv, device=torch.device('cpu')):
        super().__init__()

        self.device = device

        self.q = conv(conv_sizes=conv_sizes, dense_sizes=dense_sizes, activation=activation_conv)
        self.q_last = conv_last(out_size=1, activation=nn.Tanh, input_size=dense_sizes[-1]+2+2)

    def forward(self, field, agent_center, act):
        field = torch.as_tensor(field).to(self.device)
        if len(field.shape) == 3:
            field = field.unsqueeze(0)

        emb = self.q(field)

        if len(agent_center.shape) == 1:
            agent_center = agent_center.unsqueeze(0)

        emb = torch.concat([emb, agent_center], dim=1)

        q = self.q_last(torch.cat([emb, act], dim=-1))
        return torch.squeeze(q, -1) # Critical to ensure q has right shape.


class MLPActorCritic(nn.Module):

    def __init__(self, device, conv_sizes=(32, 64, 128), dense_sizes=(512, 128), activation_conv=nn.ReLU):
        super().__init__()

        self.device = device

        act_limit = 1

        # build policy and value functions
        self.pi = SquashedGaussianMLPActor(act_limit, conv_sizes=conv_sizes, dense_sizes=dense_sizes, activation_conv=activation_conv, device=device).to(device)
        self.q1 = MLPQFunction(conv_sizes=conv_sizes, dense_sizes=dense_sizes, activation_conv=activation_conv, device=device).to(device)
        self.q2 = MLPQFunction(conv_sizes=conv_sizes, dense_sizes=dense_sizes, activation_conv=activation_conv, device=device).to(device)
        with torch.no_grad():
            init_weights(self.pi)
            init_weights(self.q1)
            init_weights(self.q2)

    def act(self, obs, deterministic=False):
        field = torch.as_tensor(obs['world'], dtype=torch.float32).to(self.device)
        position = torch.as_tensor(obs['agent_center'], dtype=torch.float32).to(self.device) / 1536

        with torch.no_grad():
            a, _ = self.pi(field, position, deterministic, False)

            return a.cpu().numpy()
