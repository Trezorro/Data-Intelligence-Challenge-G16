import spinup.algos.pytorch.ppo.ppo as ppo
import torch
import torch.nn as nn
import gym
from gym.envs.registration import register
from gym import Env

register(
    id="ContinuousWorld-v0",
    entry_point="simulation.environment:ContinuousEnv"
)

env_fn: Env =  lambda : gym.make("ContinuousWorld-v0")
# env_fn = lambda : gym.make('LunarLander-v2')

ac_kwargs = dict(hidden_sizes=[64,64], activation=torch.nn.ReLU)

logger_kwargs = dict(output_dir='../spinningup/test/path/to/output_dir', exp_name='experiment_name')

ppo.ppo(env_fn=env_fn, ac_kwargs=ac_kwargs, steps_per_epoch=5000, epochs=20, logger_kwargs=logger_kwargs)

# import gym
#
# from stable_baselines.common.policies import MlpPolicy
# from stable_baselines import PPO1
#
# env_fn = lambda : gym.make('LunarLander-v2')
#
# model = PPO1(MlpPolicy, env, verbose=1)
# model.learn(total_timesteps=25000)
# model.save("ppo1_cartpole")
#
# del model # remove to demonstrate saving and loading
#
# model = PPO1.load("ppo1_cartpole")
#
# obs = env.reset()
# while True:
#     action, _states = model.predict(obs)
#     obs, rewards, dones, info = env.step(action)
#     env.render()