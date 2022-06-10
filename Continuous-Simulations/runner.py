import gym
from gym.envs.registration import register
from tqdm import trange
from gym import Env
from agent_configs.random_agent import RandomAgent

register(
    id="ContinuousWorld-v0",
    entry_point="simulation.environment:ContinuousEnv"
)

options = {
    "battery_drain": 0.1,
    "agent_width": 96
}

if __name__ == '__main__':
    render_mode = "human"
    env: Env = gym.make("ContinuousWorld-v0")
    observation, info = env.reset(seed=42, return_info=True, options=options)
    agent = RandomAgent()
    try:
        for _ in trange(2000):
            env.render(mode=render_mode)
            # WIP FIX YVAN reset and step give observation
            action = agent.step(observation)
            observation, reward, done, info = env.step(action=action)

            if done:
                observation, info = env.reset(seed=42, return_info=True,
                                              options=options)
    finally:
        env.close()
