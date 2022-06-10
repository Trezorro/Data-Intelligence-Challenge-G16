import gym
from gym.envs.registration import register
from tqdm import trange

register(
    id="ContinuousWorld-v0",
    entry_point="simulation.environment:ContinuousEnv"
)

if __name__ == '__main__':
    render_mode = "human"
    env = gym.make("ContinuousWorld-v0")
    observation, info = env.reset(seed=42, return_info=True)
    try:
        for _ in trange(500):
            env.render(mode=render_mode)
            # WIP FIX YVAN reset and step give observation
            action = env.action_space.sample()
            observation, reward, done, info = env.step(action=action)

            if done:
                observation, info = env.reset(seed=42, return_info=True)
    finally:
        env.close()
