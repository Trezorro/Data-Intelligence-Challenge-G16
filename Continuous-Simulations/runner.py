from simulation.environment import ContinuousEnv
from gym.envs.registration import register
import gym

register(
    id="ContinuousWorld-v0",
    entry_point="simulation.environment:ContinuousEnv"
)

if __name__ == '__main__':
    env = gym.make("ContinuousWorld-v0")
    env.reset(seed=42)

    for _ in range(500):
        env.render()
    env.close()
