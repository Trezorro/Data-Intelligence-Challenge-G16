from baseline_agent.random_agent import get_action_random_agent
from spinup.utils.test_policy import load_policy_and_env, run_policy

import gym
from gym.envs.registration import register

register(
    id="ContinuousWorld-v0",
    entry_point="simulation.environment:ContinuousEnv"
)

# _, get_action = load_policy_and_env("./data")
get_action = get_action_random_agent      # Random agent -- Baseline
env = gym.make('ContinuousWorld-v0')
run_policy(env, get_action, num_episodes=3, max_ep_len=900)
