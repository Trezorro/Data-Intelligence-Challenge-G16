import logging
from typing import Tuple, Dict
from tqdm import tqdm

from environment import Robot, Grid
from helpers.reward_functions import get_label_and_battery_based_reward
import numpy as np
import copy

logger = logging.getLogger(__name__)

class SarsaState:
    def __init__(self, pos_x: int, pos_y: int, vision: dict):
        """State for Sarsa Lookup.

        Vision dict should have keys "n", "e", "w", "s" for which the values
        are True if clean and False if dirty. Walls and obstacles are always
        True.
        """
        self.pos_x = pos_x
        self.pos_y = pos_y
        self.vision = vision

    def get_index(self, action):
        """Get index of Q table for Sarsa given for this state.

        The Q table has 4 dimensions. The first 2 are the physical grid, the 3rd
        dimension is the combination of all possible vision states (len = 16),
        and the 4th dimension is the possible actions.
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
        i = action_map[action] if action is not None else None

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

        self.starting_pos = copy.deepcopy(pos)
        self.starting_grid = copy.deepcopy(grid)
        self.starting_orientation = copy.copy(orientation)

        self.Q = np.zeros((grid.n_rows, grid.n_cols, 2**4, 4))

        self.is_trained = False

    def reset_env(self):
        self.grid = copy.deepcopy(self.starting_grid)
        self.pos = copy.deepcopy(self.starting_pos)
        self.orientation = copy.copy(self.starting_orientation)

        self.history = [[self.pos[0]], [self.pos[1]]]
        self.alive = True
        self.battery_lvl = 100

    def do_move(self):
        directions = ["n", "e", "s", "w"]
        current_state = SarsaState(self.pos[1], self.pos[0], self.get_vision())
        x, y, z, _ = current_state.get_index(None)
        action_idx = np.argmax(self.Q[(x, y, z)])
        action = directions[action_idx]

        # Rotate
        while action != self.orientation:
            self.rotate('r')

        self.move()

    def get_vision(self) -> Dict:
        d = {'n': False, 'e': False, 's': False, 'w': False}

        for dir in ['n', 'e', 's', 'w']:
            pos = tuple(np.array(self.pos) + np.array(self.dirs[dir]))
            val = self.grid.get_c(pos)
            if -2 <= val <= 0:
                d[dir] = True

        return d

    def train(self):
        logger.info("Sarsa.train: Started training robot for " + str(self.number_of_episodes) + " iterations.")

        for episode in tqdm(range(self.number_of_episodes)):
            t = 0
            self.reset_env()

            state: SarsaState = SarsaState(self.pos[1], self.pos[0], self.get_vision())
            action = self.choose_action(state)

            while t < self.max_steps_per_episode:
                new_state, reward, done = self.step(action)

                new_action = self.choose_action(new_state)

                self.update(state, action, reward, new_state, new_action)

                state = new_state.make_copy()
                action = tuple(*new_action)

                t += 1

                if done:
                    break

        self.is_trained = True

    def step(self, action: str) -> Tuple[SarsaState, float, bool]:
        # Rotate
        while action != self.orientation:
            self.rotate('r')

        _, drained_battery = self.move()

        reward = get_label_and_battery_based_reward(self.grid.get_c(self.pos), drained_battery)

        new_state = SarsaState(self.pos[1], self.pos[0], self.get_vision())

        done = not (self.alive or self.battery_lvl > 0) or self.grid.is_cleaned()

        return new_state, reward, done

    def choose_action(self, current_state) -> str:
        directions = ["n", "e", "s", "w"]
        if np.random.uniform(0, 1) < self.epsilon:
            action = self.dirs[np.random.choice(directions)]
        else:
            x, y, z, _ = current_state.get_index(None)
            action_idx = np.argmax(self.Q[(x, y, z)])
            action = directions[action_idx]

        return action

    def update(self, state_1, action_1, reward, state_2, action_2):
        x1, y1, z1, i1 = state_1.get_index(action_1)
        x2, y2, z2, i2 = state_2.get_index(action_2)

        predict = self.Q[x1, y1, z1, i1]
        target = reward + self.gamma * self.Q[x2, y2, z2, i2]

        update_coef = self.lr * (target - predict)
        self.Q[x1, y1, z1, i1] = self.Q[x1, y1, z1, i1] + update_coef


def robot_epoch(robot: Sarsa):
    robot.do_move()
