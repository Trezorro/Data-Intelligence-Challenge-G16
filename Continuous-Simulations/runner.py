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
        # WIP FIX YVAN reset and step give observation
        action = env.action_space.sample() # TODO: Give agent the state and ask for move
        env.step(action=action)
        env.render()
    env.close()
