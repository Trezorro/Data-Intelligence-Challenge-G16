import logging
from typing import Tuple, Dict
from random import randint

from environment import Robot, Grid
from helpers.reward_functions import get_label_and_battery_based_reward
from helpers.td_state import TDState
import numpy as np
import copy

logger = logging.getLogger(__name__)


class TDAgent(Robot):

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
        self.starting_grid = grid  # Grid linked to the visualization, so no deepcopy is made as its needed later
        self.starting_orientation = copy.copy(orientation)

        # Initialize Q table
        self.Q = np.zeros((grid.n_rows, grid.n_cols, 2**4, 4))

        self.is_trained = False

    def train(self) -> None:
        """ Trains the robot according to the specified TD algorithm.

        Uses the other methods and the given parameters upon initialization to train the robot using the TD
        algorithm with decreasing learning rate and epsilon exploration.
        """

    def do_move(self) -> None:
        """ Function executes a move according to the robot's Q_table
        """

        # Check if robot is trained.
        if not self.is_trained:
            logger.warning("TD.do_move: Executing robot move without being trained!")

        # Get action according to TD policy
        directions = ["n", "e", "s", "w"]
        current_state = TDState(self.pos[1], self.pos[0], self.__get_vision())
        y, x, z, _ = current_state.get_index(None)
        action_idx = np.argmax(self.Q[(y, x, z)])
        action = directions[action_idx]

        # Rotate bot in correct direction
        while action != self.orientation:
            self.rotate('r')

        # Move robot
        self.move()

    def __step(self, action: str) -> Tuple[TDState, float, bool]:
        """ This function simulates a step of the algorithm given an action and the current state of the robot.

        Args:
             action:    The action to be taken from ['n', 'e', 's', 'w']

        Returns:
            Tuple containing the following 3 items:
                new_state:  The new TDState of the robot
                reward:     The reward that was obtained for doing the move
                done:       Whether the simulation is over, based on whether the robot is alive, there is battery left
                                and whether the grid is cleaned.
        """

        # Rotate the bot in the correct direction it wants to move
        while action != self.orientation:
            self.rotate('r')

        label_square_in_front = self.grid.get_c(tuple(np.array(self.pos) + np.array(self.dirs[action])))

        _, drained_battery = self.move()

        reward = get_label_and_battery_based_reward(label_square_in_front, drained_battery)

        new_state = TDState(self.pos[1], self.pos[0], self.__get_vision())

        done = not (self.alive and self.battery_lvl > 0) or self.grid.is_cleaned()

        return new_state, reward, done

    def __choose_action(self, current_state: TDState, use_greedy_strategy: bool = False) -> str:
        """ Function chooses and action based on the epsilon greedy strategy.

        Args:
            current_state: Current TDState object

        Returns:
            The action to be taken from ['n', 'e', 's', 'w']
        """
        directions = ["n", "e", "s", "w"]
        if not use_greedy_strategy and np.random.uniform(0, 1) < self.epsilon:
            action = np.random.choice(directions)
        else:
            y, x, z, _ = current_state.get_index(None)
            action_idx = np.argmax(self.Q[(y, x, z)])
            action = directions[action_idx]

        return action

    def __update(self) -> None:
        """ Updates the Q table given several parameters.
        """
        raise NotImplementedError("Update strategy not defined. Please define your __update function")

    def __get_vision(self) -> Dict:
        """ Function retrieves the current vision of the robot according to what the TDState object expects.

        Returns:
            Dictionary according to the docs of TDState.
        """
        d = {'n': False, 'e': False, 's': False, 'w': False}

        for direction in ['n', 'e', 's', 'w']:
            pos = tuple(np.array(self.pos) + np.array(self.dirs[direction]))
            val = self.grid.get_c(pos)
            if -2 <= val <= 0:
                d[direction] = True

        return d

    def __reset_env(self, starting_position=None):
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

    def __get_random_start_pos(self) -> Tuple[int, int]:
        """ Function generates a random starting position out of the clean and dirty squares in the current grid.

        Returns:
            Tuple (y,x) representing a random starting position for the robot. This starting position is either
            clean or dirty.
        """
        while True:
            rand_x = randint(1, self.grid.n_cols-1)
            rand_y = randint(1, self.grid.n_rows-1)

            val = self.grid.get(rand_x, rand_y)

            if val == 1 or val == 0:
                return rand_y, rand_x
