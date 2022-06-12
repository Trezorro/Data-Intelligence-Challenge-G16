import gym
from gym.envs.registration import register
from tqdm import trange
from gym import Env
from agent_configs.random_agent import RandomAgent
from torch import nn
from spinup import ppo_pytorch as ppo
from spinup import sac_pytorch as sac

register(
    id="ContinuousWorld-v0",
    entry_point="simulation.environment:ContinuousEnv"
)

options = {
    "battery_drain": 0.1,
    "agent_width": 96
}

ac_kwargs = dict(hidden_sizes=[64, 64], activation=nn.ReLU)
logger_kwargs = dict(output_dir='./data', exp_name='test')
env_fn = lambda: gym.make('ContinuousWorld-v0')


ppo(env_fn=env_fn, ac_kwargs=ac_kwargs, steps_per_epoch=200, epochs=20, logger_kwargs=logger_kwargs)