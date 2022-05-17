import logging
from tqdm import tqdm

from helpers.td_robot import TDRobotBase
from helpers.td_state import TDState
import numpy as np

logger = logging.getLogger(__name__)


class Robot(TDRobotBase):
    """Q-Learning Robot"""
    
    def train(self) -> None:
        """ Trains the robot according to the QAgent algorithm.

        Uses the other methods and the given parameters upon initialization to train the robot using the QAgent
        algorithm with decreasing learning rate and epsilon exploration.
        """

        for _ in tqdm(range(self.number_of_episodes)):
            # Reset environment. There is a chance that it randomizes the starting position.
            if np.random.binomial(1, 0.2) == 1:
                self.reset_env(self.get_random_start_pos())
            else:
                self.reset_env()

            # Get initial state and action
            state = TDState(self.pos[1], self.pos[0], self.get_vision())

            for t in range(self.max_steps_per_episode):
                action = self._choose_action(state)
                # Make step given current action
                new_state, reward, done = self._step(action)

                if done:
                    max_next_reward = 0
                else:
                    # Choose a new action based on the greedy strategy
                    greedy_action = self._choose_action(new_state, use_greedy_strategy=True)
                    max_next_reward = self.Q[new_state.get_index(greedy_action)]

                # Update Q table
                self._update_qtable(state, action, reward, max_next_reward) # FIXME: possible bug

                # Break if simulation is finished
                if done:
                    break
                else:
                    # Copy over new state and new action for next iteration
                    state = new_state.make_copy()

            # Slowly lower the learning rate and epsilon exploration
            self.epsilon *= 0.9995
            self.lr *= 0.9995

        # Reset environment after training for simulation.
        self.reset_env()
        self.grid = self.starting_grid

        # Set is_trained to true after completion of training
        self.is_trained = True

    def _choose_action(self, current_state: TDState, use_greedy_strategy: bool = False) -> str:
        """Return a next action based on the epsilon greedy strategy.

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

    def _update_qtable(self, state_1, action_1, reward, max_next_q) -> None:
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
        if new_q > 500:
            logger.debug("Q value of state %s is too high: %f", current_state_idx, new_q)
        self.Q[current_state_idx] = new_q