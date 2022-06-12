from spinup.utils.test_policy import load_policy_and_env, run_policy

import gym
from gym.envs.registration import register

register(
    id="ContinuousWorld-v0",
    entry_point="simulation.environment:ContinuousEnv"
)

options = {
    "battery_drain": 0.1,
    "agent_width": 96
}

_, get_action = load_policy_and_env("./data")
env = gym.make('ContinuousWorld-v0')
run_policy(env, get_action)