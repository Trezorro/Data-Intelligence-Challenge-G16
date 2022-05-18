import logging
from typing import Tuple, Dict
from random import randint

from helpers.globals import DEBUG
from environment import RobotBase, Grid
from helpers.reward_functions import get_label_and_battery_based_reward
from helpers.td_state import TDState
import numpy as np
import copy
import abc

logger = logging.getLogger(__name__)


class TDRobotBase(RobotBase):

    def __init__(self, grid: Grid, pos: Tuple[int, int], orientation: str,
                 p_move: int = 0, battery_drain_p: int = 1,
                 battery_drain_lam: float = 1, vision: int = 1,
                 epsilon: float = 0.99, gamma: float = 0.95, lr: float = 0.99,
                 max_steps_per_episode: int = 100,
                 number_of_episodes: int = 2000, train_instantly: bool = True):
        """Base class for time temporal difference based algorithms.

        Args:
            grid: The grid the robot will move on.
            pos: Current position as a tuple (y, x).
            orientation: Current orientation of the robot. One of "n", "e", "s",
                "w".
            p_move: Probability of robot performing a random move instead of
                listening to a given command.
            battery_drain_p: Probability of a battery drain event happening at
                each move.
            battery_drain_lam: Amount (lambda) of battery drain (X) in
                X ~ Exponential(lambda)
            vision: Number of tiles in each of the 4 directions included in the
                robots vision.
            epsilon: epsilon parameter for the algorithm.
            gamma: Gamma parameter for the algorithm.
            lr: Learning rate for the algorithm.
            max_steps_per_episode: Number of maximum steps to train for in one
                episode.
            number_of_episodes: Number of episodes to train for.
            train_instantly: Whether to train immediately on placing the robot
                on the grid/instantiation.
        """
        # NOTE: The battery drain params are set  here, but note that if you
        # have the UI, those settings prevail (unless you comment them out in
        # app.py line 187)

        super().__init__(grid, pos, orientation, p_move, battery_drain_p,
                         battery_drain_lam, vision)

        self.epsilon = epsilon
        self.gamma = gamma
        self.lr = lr
        self.max_steps_per_episode = max_steps_per_episode
        self.number_of_episodes = number_of_episodes

        self.starting_pos = pos
        # Grid linked to the visualization, so no deepcopy is made as its
        # needed later
        self.starting_grid = grid
        self.starting_orientation = copy.copy(orientation)

        # Initialize Q table
        self.Q = np.zeros((grid.n_rows, grid.n_cols, 2**4, 4))

        self.is_trained = False
        self.show_debug_values = DEBUG
        if train_instantly:
            self.train()

    def robot_epoch(self):
        self.do_move()

    @abc.abstractmethod
    def train(self) -> None:
        """ Trains the robot according to the specified TD algorithm.

        Uses the other methods and the given parameters upon initialization to
        train the robot using the TD algorithm with decreasing learning rate and
        epsilon exploration.
        """
        raise NotImplementedError("Subclass and implement train() method!")

    def do_move(self) -> None:
        """ Function executes a move according to the robot's Q_table
        """

        # Check if robot is trained.
        if not self.is_trained:
            logger.warning("TD.do_move: Executing robot move without being "
                           "trained!")

        # Get action according to TD policy
        directions = ["n", "e", "s", "w"]
        current_state = TDState(self.pos[1], self.pos[0], self.get_vision())
        y, x, z, _ = current_state.get_index(None)
        action_idx = np.argmax(self.Q[(y, x, z)])
        action = directions[action_idx]

        # Rotate bot in correct direction
        while action != self.orientation:
            self.rotate('r')

        # Move robot
        self.move()

        if DEBUG:
            current_state = TDState(self.pos[1], self.pos[0], self.get_vision())
            for d in directions:
                target_pos = tuple(np.array(self.pos) + np.array(self.dirs[d]))
                q_idx = current_state.get_index(d)
                self.debug_values[target_pos] = self.Q[q_idx]

    def _step(self, action: str) -> Tuple[TDState, float, bool]:
        """ This function simulates a step of the algorithm.

        This is done given an action and the current state of the robot.

        Args:
             action:    The action to be taken from ['n', 'e', 's', 'w']

        Returns:
            Tuple containing the following 3 items:
                new_state:  The new TDState of the robot
                reward:     The reward that was obtained for doing the move
                done:       Whether the simulation is over, based on whether the
                            robot is alive, there is battery left and whether
                            the grid is cleaned.
        """

        # Rotate the bot in the correct direction it wants to move
        while action != self.orientation:
            self.rotate('r')

        label_square_in_front = self.grid.get_c((np.array(self.pos)
                                                 + np.array(self.dirs[action])))

        _, drained_battery = self.move()

        reward = get_label_and_battery_based_reward(label_square_in_front,
                                                    drained_battery)

        new_state = TDState(self.pos[1], self.pos[0], self.get_vision())

        done = (not (self.alive and self.battery_lvl > 0)
                or self.grid.is_cleaned())

        return new_state, reward, done

    def get_vision(self) -> Dict:
        """ Function retrieves the current vision of the robot.

        This is done according to what the TDState object expects.

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

    def reset_env(self, starting_position=None):
        """ Function resets the environment for the next simulation.

        Args:
            starting_position: the new starting position of the robot. If not
                included, default starting position upon initialization is
                taken.
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

    def get_random_start_pos(self) -> Tuple[int, int]:
        """ Function generates a random starting position.

        The random starting position is out of the clean and dirty squares in
        the current grid.

        Returns:
            (y,x) representing a random starting position for the robot. This
                starting position is either clean or dirty.
        """
        while True:
            rand_x = randint(1, self.grid.n_cols-1)
            rand_y = randint(1, self.grid.n_rows-1)

            val = self.grid.get(rand_x, rand_y)

            if val == 1 or val == 0:
                return rand_y, rand_x
