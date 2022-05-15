import logging
from typing import Tuple, Dict
from tqdm import tqdm
from random import randint

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

        The Q table has 4 dimensions. The first 2 are the physical grid (indexed y,x), the 3rd
        dimension is the combination of all possible vision states (len = 16),
        and the 4th dimension is the possible actions.
        """
        y = self.pos_y
        x = self.pos_x
        z = self.vision["n"] * 1 \
            + self.vision["e"] * 2 \
            + self.vision["s"] * 4 \
            + self.vision["w"] * 8

        action_map = {"n": 0,
                      "e": 1,
                      "s": 2,
                      "w": 3}
        i = action_map[action] if action is not None else None

        return y, x, z, i

    def make_copy(self):
        return SarsaState(self.pos_x, self.pos_y, self.vision)


class Sarsa(Robot):

    def __init__(self, grid: Grid, pos, orientation, p_move=0, battery_drain_p=1, battery_drain_lam=1, vision=1,
                 epsilon=0.99, gamma=0.95, lr=0.99, max_steps_per_episode=100, number_of_episodes=5000):
        # NOTE: i have set the battery drain params here, but note that if you have the UI, those settings
        # prevail (unless you comment them out in app.py line 187)

        super().__init__(grid, pos, orientation, p_move, battery_drain_p, battery_drain_lam, vision)

        self.epsilon = epsilon
        self.gamma = gamma
        self.lr = lr
        self.max_steps_per_episode = max_steps_per_episode
        self.number_of_episodes = number_of_episodes

        self.starting_pos = pos
        self.starting_grid = grid # This grid is linked to the visualization, so no deepcopy is made as its needed later
        self.starting_orientation = copy.copy(orientation)

        # Initialize Q table
        self.Q = np.zeros((grid.n_rows, grid.n_cols, 2**4, 4))

        self.is_trained = False

    def reset_env(self, starting_position = None):
        """ Function resets the environment for the next simulation.

        Args:
            starting_position: the new starting position of the robot. If not included, default starting position
                                upon initialization is taken.

        """
        self.grid = copy.deepcopy(self.starting_grid)

        if starting_position is None:
            self.pos = copy.deepcopy(self.starting_pos)
        else:
            self.pos = starting_position
            self.grid.put_c(self.starting_pos, 1)
            self.grid.put_c(starting_position, -6)

        self.orientation = copy.copy(self.starting_orientation)

        self.history = [[self.pos[0]], [self.pos[1]]]
        self.alive = True
        self.battery_lvl = 100

    def get_vision(self) -> Dict:
        d = {'n': False, 'e': False, 's': False, 'w': False}

        for dir in ['n', 'e', 's', 'w']:
            pos = tuple(np.array(self.pos) + np.array(self.dirs[dir]))
            val = self.grid.get_c(pos)
            if -2 <= val <= 0:
                d[dir] = True

        return d

    def get_random_start_pos(self):
        while True:
            randx = randint(1, self.grid.n_cols-1)
            randy = randint(1, self.grid.n_rows-1)

            val = self.grid.get(randx, randy)

            if val == 1 or val == 0:
                return randy, randx

    def train(self):
        logger.info("Sarsa.train: Started training robot for " + str(self.number_of_episodes) + " iterations.")

        for iter in tqdm(range(self.number_of_episodes)):
            t = 0
            if np.random.binomial(1, 0.2) == 1:
                self.reset_env(self.get_random_start_pos())
            else:
                self.reset_env()

            state: SarsaState = SarsaState(self.pos[1], self.pos[0], self.get_vision())
            action = self.choose_action(state)

            while t < self.max_steps_per_episode:
                new_state, reward, done = self.step(action)

                new_action = self.choose_action(new_state)

                self.update(state, action, reward, new_state, new_action)

                state = new_state.make_copy()  # copy
                action = str(new_action)  # copy

                t += 1

                if done:
                    break

            # Slowly lower the learning rate and epsilon exploration
            self.epsilon *= 0.9995
            self.lr *= 0.9995

            if iter % 100 == 0:
                logger.info("")
                logger.info(str(self.lr))
                logger.info(iter) # For debugging


        self.reset_env()
        self.grid = self.starting_grid

        self.is_trained = True

    def step(self, action: str) -> Tuple[SarsaState, float, bool]:
        # Rotate
        while action != self.orientation:
            self.rotate('r')

        label_square_in_front = self.grid.get_c(tuple(np.array(self.pos) + np.array(self.dirs[action])))

        _, drained_battery = self.move()

        reward = get_label_and_battery_based_reward(label_square_in_front, drained_battery)

        new_state = SarsaState(self.pos[1], self.pos[0], self.get_vision())

        done = not (self.alive or self.battery_lvl > 0) or self.grid.is_cleaned()

        return new_state, reward, done

    def choose_action(self, current_state) -> str:
        directions = ["n", "e", "s", "w"]
        if np.random.uniform(0, 1) < self.epsilon:
            action = np.random.choice(directions)
        else:
            y, x, z, _ = current_state.get_index(None)
            action_idx = np.argmax(self.Q[(y, x, z)])
            action = directions[action_idx]

        return action

    def update(self, state_1, action_1, reward, state_2, action_2):
        index_1 = state_1.get_index(action_1)
        index_2 = state_2.get_index(action_2)

        predict = self.Q[index_1]
        target = reward + self.gamma * self.Q[index_2]

        update_coef = self.lr * (target - predict)
        self.Q[index_1] = self.Q[index_1] + update_coef


def robot_epoch(robot: Sarsa):
    directions = ["n", "e", "s", "w"]
    current_state = SarsaState(robot.pos[1], robot.pos[0], robot.get_vision())
    y, x, z, _ = current_state.get_index(None)
    action_idx = np.argmax(robot.Q[(y, x, z)])
    action = directions[action_idx]

    # Rotate
    while action != robot.orientation:
        robot.rotate('r')

    robot.move()
