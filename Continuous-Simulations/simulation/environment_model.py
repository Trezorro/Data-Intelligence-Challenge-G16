"""Environment Model.

The internal environment used to maintain an awareness of where everything is.
"""
import numpy as np


class EnvironmentModel:
    def __init__(self, map: np.ndarray):
        """A very simple environment model which holds data about the world.

        Args:
            map: A 24x24 array containing where walls are.
        """
        self.map = map
        self.agent_loc = self.place_agent_randomly()

    def place_agent_randomly(self):
        agent_placed = False
        while not agent_placed:
            # Keep trying until a position that is not a wall is found.
            position = np.random.randint(0, self.map.shape[0], 2)
            agent_placed = self.map[tuple(position)] == 0
        return position.astype(float)
