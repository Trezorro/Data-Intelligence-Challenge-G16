import logging
from tqdm import tqdm

from environment import Grid
from helpers.td_robot import TDRobot
from helpers.td_state import TDState
import numpy as np
import copy

logger = logging.getLogger(__name__)

DEBUG = True


class QAgent(TDRobot):

    def __init__(self, grid: Grid, pos, orientation, p_move=0, battery_drain_p=1, battery_drain_lam=1, vision=1,
                 epsilon=0.49, gamma=0.95, lr=0.99, max_steps_per_episode=100, number_of_episodes=1000):
        # NOTE: i have set the battery drain params here, but note that if you have the UI, those settings
        # prevail (unless you comment them out in app.py line 187)

        super().__init__(grid, pos, orientation, p_move, battery_drain_p, battery_drain_lam, vision,
                         epsilon, gamma, lr, max_steps_per_episode, number_of_episodes)

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
        self.show_debug_values = DEBUG

    def train(self) -> None:
        """ Trains the robot according to the QAgent algorithm.

        Uses the other methods and the given parameters upon initialization to train the robot using the QAgent
        algorithm with decreasing learning rate and epsilon exploration.
        """

        for _ in tqdm(range(self.number_of_episodes)):
            # Reset environment. There is a chance that it randomizes the starting position.
            if np.random.binomial(1, 0.2) == 1:
                self.__reset_env(self.__get_random_start_pos())
            else:
                self.__reset_env()

            # Get initial state and action
            state: TDState = TDState(self.pos[1], self.pos[0], self.__get_vision())
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

    def __update(self, state_1, action_1, reward, max_next_q) -> None:
        """ Function updates the Q table given several parameters.

        Args:
            state_1:    The first QAgentState object.
            action_1:   The action chosen from state_1 which was completed.
            reward:     The reward obtained from doing action_1 in state_1.
            max_next_q: The max Q reachable from the state after the action
        """
        current_state_idx = state_1.get_index(action_1)

        current_q = self.Q[current_state_idx]
        delta = reward + self.gamma * (max_next_q - current_q)
        new_q = current_q + self.lr * delta
        self.Q[current_state_idx] = new_q


def robot_epoch(robot: QAgent):
    robot.do_move()
