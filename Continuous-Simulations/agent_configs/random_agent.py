import numpy as np


class RandomAgent:
    @staticmethod
    def step(observation: dict) -> dict:
        """Changes to a random direction on hitting a wall."""
        if observation["hit_wall"] + observation["hit_obstacle"] > 0:
            new_dir = (np.random.random([1]) * 2) - 1,
        else:
            new_dir = np.zeros([1])
        return {"direction": new_dir, "move": 1}
