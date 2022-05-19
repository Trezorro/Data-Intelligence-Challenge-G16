import logging
from typing import Tuple, Dict
from random import randint

from helpers.globals import DEBUG
from environment import RobotBase, Grid
from helpers.reward_functions import get_label_and_battery_based_reward, get_reward_turning_off
from helpers.td_state import TDState
import numpy as np
import copy

logger = logging.getLogger(__name__)


class TDRobotBase(RobotBase):

    def __init__(self, grid: Grid, pos, orientation, p_move=0, battery_drain_p=1, battery_drain_lam=1, vision=1,
                 epsilon=0.99, gamma=0.8, lr=0.99, max_steps_per_episode=100, number_of_episodes=2000, train_instantly=True):
        # NOTE: i have set the battery drain params here, but note that if you have the UI, those settings
        # prevail (unless you comment them out in app.py line 187)

        super().__init__(grid, pos, orientation, p_move, battery_drain_p, battery_drain_lam, vision)

        self.initial_epsilon = epsilon
        self.epsilon = epsilon
        self.gamma = gamma
        self.initial_learning_rate = lr
        self.lr = lr
        self.max_steps_per_episode = max_steps_per_episode
        self.number_of_episodes = number_of_episodes

        self.starting_pos = pos
        self.starting_grid = grid  # Grid linked to the visualization, so no deepcopy is made as its needed later
        self.starting_orientation = copy.copy(orientation)
        self.starting_history = copy.deepcopy(self.history)
        self.starting_battery_lvl = float(self.battery_lvl)

        # Initialize Q table
        self.Q = np.zeros((grid.n_rows, grid.n_cols, 2**4, 5))

        self.is_trained = False
        self.show_debug_values = DEBUG
        if train_instantly:
            self.train()

    def robot_epoch(self):
        self.do_move()

    def train(self) -> None:
        """ Trains the robot according to the specified TD algorithm.

        Uses the other methods and the given parameters upon initialization to train the robot using the TD
        algorithm with decreasing learning rate and epsilon exploration.
        """
        raise NotImplementedError("Subclass and implement train() method!")

    def retrain(self) -> None:
        """ Retrains the robot from its current position and state from scratch (empty Q table)

        Resets various parameters such that the training function can be reused.
        """
        self.Q = np.zeros_like(self.Q)
        self.epsilon = self.initial_epsilon
        self.lr = self.initial_learning_rate

        self.number_of_episodes = 1000

        self.starting_pos = copy.deepcopy(self.pos)
        self.starting_grid = self.grid
        self.starting_orientation = str(self.orientation)

        self.starting_history = copy.deepcopy(self.history)
        self.starting_battery_lvl = float(self.battery_lvl)

        self.train()

    def do_move(self) -> None:
        """ Function executes a move according to the robot's Q_table
        """

        # Check if robot is trained.
        if not self.is_trained:
            logger.warning("TD.do_move: Executing robot move without being trained!")

        # Retrain every 10 moves
        if len(self.history[0]) % 1 == 0:
            self.retrain()

        # Get action according to TD policy
        actions = ["n", "e", "s", "w", "off"]
        current_state = TDState(self.pos[1], self.pos[0], self.get_vision())
        y, x, z, _ = current_state.get_index(None)
        action_idx = np.argmax(self.Q[(y, x, z)])
        action = actions[action_idx]

        if action == "off":
            self.alive = False
        else:
            # Rotate bot in correct direction
            while action != self.orientation:
                self.rotate('r')

            # Move robot
            self.move()

        if DEBUG:
            current_state = TDState(self.pos[1], self.pos[0], self.get_vision())
            for a in actions:
                if a == "off":
                    target_square = self.pos
                else:
                    target_square = tuple(np.array(self.pos) + np.array(self.dirs[a]))
                self.debug_values[target_square] = self.Q[current_state.get_index(a)]

    def _step(self, action: str) -> Tuple[TDState, float, bool]:
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
        if action == "off":
            reward = get_reward_turning_off(self)
            self.alive = False
            done = True
            new_state = TDState(self.pos[1], self.pos[0], self.get_vision())
            return new_state, reward, done

        # Rotate the bot in the correct direction it wants to move
        while action != self.orientation:
            self.rotate('r')

        label_square_in_front = self.grid.get_c(tuple(np.array(self.pos) + np.array(self.dirs[action])))

        _, drained_battery = self.move()

        reward = get_label_and_battery_based_reward(label_square_in_front, drained_battery)

        new_state = TDState(self.pos[1], self.pos[0], self.get_vision())

        done = not (self.alive and self.battery_lvl > 0) or self.grid.is_cleaned()

        return new_state, reward, done

    def get_vision(self) -> Dict:
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

    def reset_env(self, starting_position=None):
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

        self.history = copy.deepcopy(self.starting_history)
        self.alive = True
        self.battery_lvl = float(self.starting_battery_lvl)

    def get_random_start_pos(self) -> Tuple[int, int]:
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
