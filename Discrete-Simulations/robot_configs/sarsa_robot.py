import logging
from tqdm import tqdm

from environment import Grid
from helpers.td_robot import TDRobotBase
from helpers.td_state import TDState
import numpy as np

logger = logging.getLogger(__name__)


class Robot(TDRobotBase):
    """Sarsa Robot"""

    def __init__(self, grid: Grid, pos, orientation, p_move=0, battery_drain_p=1, battery_drain_lam=1, vision=1,
                 epsilon=0.99, gamma=0.8, lr=0.99, max_steps_per_episode=800, number_of_episodes=1000,
                 train_instantly=True, stop_lr=0.1, stop_eps=0.1):

        self.lr_decrease_factor = (stop_lr / lr) ** (1 / number_of_episodes)
        self.epsilon_decrease_factor = (stop_eps / epsilon) ** (1 / number_of_episodes)

        super().__init__(grid, pos, orientation, p_move, battery_drain_p, battery_drain_lam, vision, epsilon, gamma, lr,
                         max_steps_per_episode, number_of_episodes, train_instantly)

    def train(self) -> None:
        """ Trains the robot according to the Sarsa algorithm.

        Uses the other methods and the given parameters upon initialization to train the robot using the Sarsa
        algorithm with decreasing learning rate and epsilon exploration.
        """

        for _ in tqdm(range(self.number_of_episodes)):
            # Reset environment. There is a chance that it randomizes the starting position.
            if np.random.binomial(1, 0) == 1:
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
            self.lr *= self.lr_decrease_factor
            self.epsilon *= self.epsilon_decrease_factor

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
            The action to be taken from ['n', 'e', 's', 'w', 'off']
        """
        actions = ["n", "e", "s", "w", "off"]
        if np.random.uniform(0, 1) < self.epsilon:
            action = np.random.choice(actions)
        else:
            y, x, z, _ = current_state.get_index(None)
            action_idx = np.argmax(self.Q[(y, x, z)])
            action = actions[action_idx]

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
