import gym
from gym.envs.registration import register
from tqdm import trange
from gym import Env
from agent_configs.random_agent import RandomAgent
from torch import nn
from spinup import ppo_pytorch as ppo
from spinup import sac_pytorch as sac
from spinup import vpg_pytorch as vpg
from spinup import td3_pytorch as td3

register(
    id="ContinuousWorld-v0",
    entry_point="simulation.environment:ContinuousEnv"
)

options = {
    "battery_drain": 0.1,
    "agent_width": 96
}

ac_kwargs = dict(hidden_sizes=[64, 64], activation=nn.ReLU)
logger_kwargs = dict(output_dir='./data', exp_name='exp_1')
env_fn = lambda: gym.make('ContinuousWorld-v0', render_mode="non_human")

# td3(env_fn=env_fn, ac_kwargs=ac_kwargs, steps_per_epoch=100, num_test_episodes=1, max_ep_len=100, logger_kwargs=logger_kwargs)
# vpg(env_fn=env_fn, ac_kwargs=ac_kwargs, steps_per_epoch=500, epochs=20, logger_kwargs=logger_kwargs)
# ppo(env_fn=env_fn, ac_kwargs=ac_kwargs, steps_per_epoch=100, epochs=50, clip_ratio=0.3, pi_lr=3e-2, vf_lr=1e-2, train_pi_iters=10, train_v_iters=10, logger_kwargs=logger_kwargs)
sac(env_fn=env_fn, ac_kwargs=ac_kwargs, steps_per_epoch=100, epochs=10, num_test_episodes=1, max_ep_len=100, logger_kwargs=logger_kwargs)
