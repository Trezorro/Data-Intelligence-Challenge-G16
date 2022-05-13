from ..environment import Robot, Grid
import numpy as np


class SarsaState:
    def __init__(self, pos_x: int, pos_y: int, vision: dict):
        """State for Sarsa Lookup.

        Vision dict should have keys "n", "e", "w", "s" for which the values
        are True if clean and False if dirty.
        """
        self.pos_x = pos_x
        self.pos_y = pos_y
        self.vision = vision

    def get_index(self, action):
        """Get index of Q table for Sarsa given for this state.

        The Q table has an
        """
        action_map = {"n": 0,
                      "e": 1,
                      "s": 2,
                      "w": 3}
        x = self.pos_x
        y = self.pos_y
        z = self.vision["n"] * 1 \
            + self.vision["e"] * 2 \
            + self.vision["s"] * 4 \
            + self.vision["w"] * 8
        i = action_map[action]

        return x, y, z, i

    def make_copy(self):
        return SarsaState(self.pos_x, self.pos_y, self.vision)


class Sarsa(Robot):

    def __init__(self, grid: Grid, pos, orientation, p_move=0, battery_drain_p=1, battery_drain_lam=0, vision=1,
                 epsilon=0.5, gamma=0.9, lr=0.2, max_steps_per_episode=100, number_of_episodes=50):

        super().__init__(grid, pos, orientation, p_move, battery_drain_p, battery_drain_lam, vision)

        self.epsilon = epsilon
        self.gamma = gamma
        self.lr = lr
        self.max_steps_per_episode = max_steps_per_episode
        self.number_of_episodes = number_of_episodes

        self.Q = np.zeros((grid.n_rows, grid.n_cols, 2**4))

    def do_move(self):
        pass

    def train(self):
        pass

    def choose_action(self, current_state):
        pass

    def update(self, state_1, action_1, reward, state_2, action_2):
        predict =


def robot_epoch(robot: Sarsa):
    robot.do_move()
