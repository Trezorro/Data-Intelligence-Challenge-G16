import copy
import random
from tqdm import tqdm

import numpy as np

from environment import Grid
from helpers.reward_functions import get_label_and_battery_based_reward
from helpers.td_robot import TDRobotBase


ACTIONS = ("n", "e", "s", "w")
ACTIONS_IDX = tuple([i for i in range(len(ACTIONS))])  # For future modularity


class Robot(TDRobotBase):
    """ Monte Carlo Robot """

    def __init__(self, grid: Grid, pos, orientation, p_move=0, battery_drain_p=1, battery_drain_lam=1, vision=1,
                 epsilon=0.1, gamma=0.8, lr=None, max_steps_per_episode=50, number_of_episodes=2000,
                 train_instantly=False):

        self.policy = np.full((*grid.cells.shape, len(ACTIONS)), 1. / len(ACTIONS))

        super().__init__(grid, pos, orientation, p_move, battery_drain_p, battery_drain_lam, vision, epsilon, gamma, lr,
                         max_steps_per_episode, number_of_episodes, train_instantly)

    def robot_epoch(self):
        move = ACTIONS[np.argmax(self.policy[self.pos[0], self.pos[1]])]

        while self.orientation != move:
            self.rotate('r')

        self.move()

    def train(self) -> None:
        """ Monte Carlo On Policy implementation."""

        # Holds the actual q grid
        self.Q = np.full((*self.grid.cells.shape, len(ACTIONS)), 0.)
        # Holds sums of G value for every grid position, used to assign a value to q
        g_sums = np.full((*self.grid.cells.shape, len(ACTIONS)), 0.)
        # Policy values
        self.policy = np.full((*self.grid.cells.shape, len(ACTIONS)), 1. / len(ACTIONS))

        # generate episodes until reach the max number of episodes
        for _ in tqdm(range(self.number_of_episodes)):
            # gradually reduce the epsilon parameter because we need less exploration
            # and more exploitation as the episodes increases
            single_episode = self.generate_episodes()
            g = 0

            states_and_actions = [[x[0][0], x[0][1], x[1]] for x in single_episode]

            for t, step in enumerate(reversed(single_episode)):
                g = self.gamma * g + step[2]

                step_y = step[0][0]
                step_x = step[0][1]
                episode_action_idx = step[1]

                before_step = states_and_actions[:len(single_episode) - t - 1]

                if [step_y, step_x, episode_action_idx] not in before_step:

                    # update returns(state,action) & q_grid(state,action)
                    g_sums[step_y, step_x, episode_action_idx] += g
                    self.Q[step_y, step_x, episode_action_idx] = g_sums[step_y, step_x, episode_action_idx] / (t + 1)

                    # calculate the best action
                    best_action_idx = np.argmax(self.Q[step_y, step_x])

                    # update the policy matrix of the specific (state,action) pair
                    # based on e-soft policy
                    for action_idx in range(len(ACTIONS)):  # enumerate action space
                        if action_idx == best_action_idx:
                            self.policy[step_y, step_x, action_idx] = 1 - self.epsilon + self.epsilon / len(ACTIONS)
                        else:
                            self.policy[step_y, step_x, action_idx] = self.epsilon / len(ACTIONS)

    def generate_episodes(self):
        """ Generate the episodes based on the policy-action probabilities.

        Returns a list called episodes that includes the state, the action and the reward of the next action i.e.
         [[(1, 1), (1, 0), 2], [(2, 1), (1, 0), -1], [(3, 1), (-1, 0), -1]]: the state (1,1) has next action (1,0)
         that gives 2 as reward, then, the state (2,1) has (1,0) as next action that gives -1 as reward etc.
        """
        # create a deepcopy of robot object to run in the simulations
        temp_robot = copy.deepcopy(self)

        episodes = []

        for step in range(self.max_steps_per_episode):
            current_pos = tuple(temp_robot.pos)
            chosen_action_idx = random.choices(ACTIONS_IDX,
                                               weights=self.policy[current_pos[0], current_pos[1]],
                                               k=1)[0]
            new_pos = tuple(np.asarray(current_pos)
                            + self.dirs[ACTIONS[chosen_action_idx]])
            label = temp_robot.grid.get_c(new_pos)

            # update position of the simulated robot
            while temp_robot.orientation != ACTIONS[chosen_action_idx]:
                temp_robot.rotate("r")

            _, battery_drained = temp_robot.move()

            reward = get_label_and_battery_based_reward(label, battery_drained)
            episodes.append([current_pos, chosen_action_idx, reward])

            if not temp_robot.alive or temp_robot.grid.is_cleaned():
                break

        return episodes
