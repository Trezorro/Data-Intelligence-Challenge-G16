import gym
from gym.envs.registration import register
from tqdm import trange
from simulation.environment import ContinuousEnv

register(
    id="ContinuousWorld-v0",
    entry_point="simulation.environment:ContinuousEnv"
)

if __name__ == '__main__':
    render_mode = "human"
    env: ContinuousEnv = gym.make("ContinuousWorld-v0")
    observation, info = env.reset(seed=42, return_info=True)
    try:
        for _ in trange(2000):
            env.render(mode=render_mode)
            # WIP FIX YVAN reset and step give observation
            action = env.action_space.sample()
            action['direction'] /= 5
            action['move'] =1
            observation, reward, done, info = env.step(action=action)

            if done:
                observation, info = env.reset(seed=42, return_info=True)
    finally:
        env.close()
