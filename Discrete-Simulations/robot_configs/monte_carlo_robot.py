import copy
import random

import numpy as np

from environment import RobotBase
from helpers.reward_functions import get_label_and_battery_based_reward
from tqdm import tqdm


ACTIONS = ("n", "e", "s", "w")
ACTIONS_IDX = tuple([i for i in range(len(ACTIONS))])  # For future modularity


def generate_episodes(policy: np.ndarray, robot: RobotBase):
    """ Generate the episodes based on the policy-action probabilities.

    Args:
        policy: The policy array to be used to generate episodes to train on.
        robot: The robot object to be used.

    Returns a list called episodes that includes the state, the action and the reward of the next action i.e.
     [[(1, 1), (1, 0), 2], [(2, 1), (1, 0), -1], [(3, 1), (-1, 0), -1]]: the state (1,1) has next action (1,0)
     that gives 2 as reward, then, the state (2,1) has (1,0) as next action that gives -1 as reward etc.
    """
    # create a deepcopy of robot object to run in the simulations
    temp_robot = copy.deepcopy(robot)

    episodes = []

    for step in range(100):
        current_pos = tuple(temp_robot.pos)
        chosen_action_idx = random.choices(ACTIONS_IDX,
                                       weights=policy[current_pos[0], current_pos[1]],
                                       k=1)[0]
        new_pos = tuple(np.asarray(current_pos)
                        + robot.dirs[ACTIONS[chosen_action_idx]])
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


def robot_epoch(robot: RobotBase, g=0.6, max_episodes=1000, epsilon=0.99):
    """ Monte Carlo On Policy implementation.

    Args:
        robot: The robot object to act on.
        g: Gamma value.
        max_episodes: Max episodes to train for.
        epsilon: Learning rate
    """
    # Holds the actual q grid
    q_grid = np.full((*robot.grid.cells.shape, len(ACTIONS)), 0.)
    # Holds sums of G value for every grid position, used to assign a value to q
    g_sums = np.full((*robot.grid.cells.shape, len(ACTIONS)), 0.)
    # Policy values
    policy = np.full((*robot.grid.cells.shape, len(ACTIONS)), 1. / len(ACTIONS))

    # generate episodes until reach the max number of episodes
    for _ in tqdm(range(max_episodes)):
        # gradually reduce the epsilon parameter cause we need less exploration
        # and more exploitation as the episodes increases
        epsilon *= 0.99
        single_episode = generate_episodes(policy, robot)
        G = 0

        states_and_actions = [[x[0][0], x[0][1], x[1]] for x in single_episode]

        for t, step in enumerate(reversed(single_episode)):
            G = g * G + step[2]

            step_y = step[0][0]
            step_x = step[0][1]
            episode_action_idx = step[1]

            before_step = states_and_actions[:len(single_episode) - t - 1]

            if [step_y, step_x, episode_action_idx] not in before_step:

                # update returns(state,action) & q_grid(state,action)
                g_sums[step_y, step_x, episode_action_idx] += G
                q_grid[step_y, step_x, episode_action_idx] = g_sums[step_y, step_x, episode_action_idx] / (t + 1)

                # calculate the best action
                best_action_idx = np.argmax(q_grid[step_y, step_x])

                # update the policy matrix of the specific (state,action) pair
                # based on e-soft policy
                for action_idx in range(len(ACTIONS)):  # enumerate action space
                    if action_idx == best_action_idx:
                        policy[step_y, step_x, action_idx] = 1 - epsilon + epsilon / len(ACTIONS)
                    else:
                        policy[step_y, step_x, action_idx] = epsilon / len(ACTIONS)

    # when the episodes iteration finishes
    # choose the corresponding robot action based on the updated policy probabilities
    corresponding_orientation = random.choices(
        ACTIONS, weights=policy[robot.pos[0], robot.pos[1]], k=1)[0]

    while robot.orientation != corresponding_orientation:
        robot.rotate('r')

    robot.move()
