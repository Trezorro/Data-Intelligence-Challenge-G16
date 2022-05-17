import logging
from tqdm import tqdm

from helpers.td_robot import TDRobotBase
from helpers.td_state import TDState
import numpy as np

logger = logging.getLogger(__name__)

class Robot(TDRobotBase):
    """Sarsa Robot"""
    def train(self) -> None:
        """ Trains the robot according to the Sarsa algorithm.

        Uses the other methods and the given parameters upon initialization to train the robot using the Sarsa
        algorithm with decreasing learning rate and epsilon exploration.
        """

        for _ in tqdm(range(self.number_of_episodes)):
            # Reset environment. There is a chance that it randomizes the starting position.
            if np.random.binomial(1, 0.2) == 1:
                self.reset_env(self.get_random_start_pos())
            else:
                self.reset_env()

            # Get initial state and action
            state: TDState = TDState(self.pos[1], self.pos[0], self.get_vision())
            action = self._choose_action(state)

            for t in range(self.max_steps_per_episode):
                # Make step given current action
                new_state, reward, done = self._step(action)

                # Choose a new action based on the new state
                new_action = self._choose_action(new_state)

                # Update Q table
                self._update_qtable(state, action, reward, new_state, new_action)

                # Copy over new state and new action for next iteration
                state = new_state.make_copy()  # copy
                action = str(new_action)  # copy

                # Break if simulation is finished
                if done:
                    break

            # Slowly lower the learning rate and epsilon exploration
            self.epsilon *= 0.9995
            self.lr *= 0.9995

        # Reset environment after training for simulation.
        self.reset_env()
        self.grid = self.starting_grid

        # Set is_trained to true after completion of training
        self.is_trained = True

    def _choose_action(self, current_state: TDState) -> str:
        """ Function chooses and action based on the epsilon greedy strategy.

        Args:
            current_state: Current TDState object

        Returns:
            The action to be taken from ['n', 'e', 's', 'w']
        """
        directions = ["n", "e", "s", "w"]
        if np.random.uniform(0, 1) < self.epsilon:
            action = np.random.choice(directions)
        else:
            y, x, z, _ = current_state.get_index(None)
            action_idx = np.argmax(self.Q[(y, x, z)])
            action = directions[action_idx]

        return action

    def _update_qtable(self, state_1, action_1, reward, state_2, action_2) -> None:
        """Function updates the Q table given several parameters.

        Args:
            state_1:    The first TDState object.
            action_1:   The action chosen from state_1 which was completed.
            reward:     The reward obtained from doing action_1 in state_1.
            state_2:    The TDState that was obtained by doing action_1 in state_1.
            action_2:   The action chosen from state_2.
        """
        index_1 = state_1.get_index(action_1)
        index_2 = state_2.get_index(action_2)

        predict = self.Q[index_1]
        target = reward + self.gamma * self.Q[index_2]

        update_coefficient = self.lr * (target - predict)
        self.Q[index_1] = self.Q[index_1] + update_coefficient
