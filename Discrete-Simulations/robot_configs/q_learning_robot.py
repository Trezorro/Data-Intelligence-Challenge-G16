import logging
from typing import Tuple, Dict
from tqdm import tqdm
from random import randint

from environment import Robot, Grid
from helpers.reward_functions import get_label_and_battery_based_reward
import numpy as np
import copy

logger = logging.getLogger(__name__)


class QAgentState:
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
        return QAgentState(self.pos_x, self.pos_y, self.vision)


class QAgent(Robot):

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
        """ Trains the robot according to the Sarsa algorithm.

        Uses the other methods and the given parameters upon initialization to train the robot using the Sarsa
        algorithm with decreasing learning rate and epsilon exploration.
        """

        for _ in tqdm(range(self.number_of_episodes)):
            # Reset environment. There is a chance that it randomizes the starting position.
            if np.random.binomial(1, 0.2) == 1:
                self.__reset_env(self.__get_random_start_pos())
            else:
                self.__reset_env()

            # Get initial state and action
            state: QAgentState = QAgentState(self.pos[1], self.pos[0], self.__get_vision())
            action = self.__choose_action(state)

            for t in range(self.max_steps_per_episode):
                # Make step given current action
                new_state, reward, done = self.__step(action)

                if done:
                    max_next_reward = 0
                else:
                    # Choose a new action based on the greedy strategy
                    greedy_action = self.__choose_action(new_state, choose_greedy_move=True)

                    max_next_reward = self.Q[new_state.get_index(greedy_action)]

                # Update Q table
                self.__update(state, action, reward, max_next_reward)

                # Break if simulation is finished
                if done:
                    break
                else:
                    # Copy over new state and new action for next iteration
                    state = new_state.make_copy()
                    action = str(greedy_action)

            # Slowly lower the learning rate and epsilon exploration
            self.epsilon *= 0.9995
            self.lr *= 0.9995

        # Reset environment after training for simulation.
        self.__reset_env()
        self.grid = self.starting_grid

        # Set is_trained to true after completion of training
        self.is_trained = True

    def do_move(self) -> None:
        """ Function executes a move according to the robots Q_table

        """

        # Check if robot is trained.
        if not self.is_trained:
            logger.warning("Sarsa.do_move: Executing robot move without being trained!")

        # Get action according to Sarsa policy
        directions = ["n", "e", "s", "w"]
        current_state = QAgentState(self.pos[1], self.pos[0], self.__get_vision())
        y, x, z, _ = current_state.get_index(None)
        action_idx = np.argmax(self.Q[(y, x, z)])
        action = directions[action_idx]

        # Rotate bot in correct direction
        while action != self.orientation:
            self.rotate('r')

        # Move robot
        self.move()

    def __step(self, action: str) -> Tuple[QAgentState, float, bool]:
        """ This function simulates a step of the algorithm given an action and the current state of the robot.

        Args:
             action:    The action to be taken from ['n', 'e', 's', 'w']

        Returns:
            Tuple containing the following 3 items:
                new_state:  The new SarsaState of the robot
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

        new_state = QAgentState(self.pos[1], self.pos[0], self.__get_vision())

        done = not (self.alive or self.battery_lvl > 0) or self.grid.is_cleaned()

        return new_state, reward, done

    def __choose_action(self, current_state: QAgentState, choose_greedy_move: bool = False) -> str:
        """ Function chooses and action based on the epsilon greedy strategy.

        Args:
            current_state: Current SarsaState object

        Returns:
            The action to be taken from ['n', 'e', 's', 'w']
        """
        directions = ["n", "e", "s", "w"]
        if not choose_greedy_move and np.random.uniform(0, 1) < self.epsilon:
            action = np.random.choice(directions)
        else:
            y, x, z, _ = current_state.get_index(None)
            action_idx = np.argmax(self.Q[(y, x, z)]) # How does it
            action = directions[action_idx]

        return action

    def __update(self, state_1, action_1, reward, max_next_q) -> None:
        """ Function updates the Q table given several parameters.

        Args:
            state_1:    The first SarsaState object.
            action_1:   The action chosen from state_1 which was completed.
            reward:     The reward obtained from doing action_1 in state_1.
            state_2:    The SarsaState that was obtained by doing action_1 in state_1.
            action_2:   The action chosen from state_2.
        """
        index_1 = state_1.get_index(action_1)

        current_q = self.Q[index_1]
        target = reward + self.gamma * (max_next_q - current_q)
        new_q = current_q + self.lr * target
        self.Q[index_1] = new_q


    def __get_vision(self) -> Dict:
        """ Function retrieves the current vision of the robot according to what the SarsaState object expects.

        Returns:
            Dictionary according to the docs of SarsaState.
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


def robot_epoch(robot: QAgent):
    robot.do_move()
