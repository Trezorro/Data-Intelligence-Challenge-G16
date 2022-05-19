import copy
import random

import numpy as np

from environment import Robot
from helpers.label_based_reward import get_reward
from tqdm import tqdm


ACTIONS = ("n", "e", "s", "w", "off")
ACTIONS_IDX = (0, 1, 2, 3, 4)


def generate_episodes(policy: np.ndarray, robot: Robot):
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
    position = temp_robot.pos

    for step in range(800):
        chosen_action = random.choices(ACTIONS_IDX,
                                       weights=policy[position[0], position[1]],
                                       k=1)[0]

        # update the reward in the simulated grid
        new_pos = tuple(np.asarray(position) + robot.dirs[ACTIONS[chosen_action]])
        reward = get_reward(temp_robot.grid.cells[new_pos])

        episodes.append([position, chosen_action, reward])

        # update position of the simulated robot
        position = new_pos
        temp_robot.pos = position

    return episodes


def robot_epoch(robot: Robot, g=0.99, max_episodes=100, epsilon=0.99):
    """ Monte Carlo On Policy implementation.

    Args:
        robot: The robot object to act on.
        g: Gamma value.
        max_episodes: Max episodes to train for.
        epsilon: Learning rate
    """
    max_episodes = max_episodes

    # Holds the actual q grid
    q_grid = np.full((robot.grid.n_rows, robot.grid.n_cols, len(ACTIONS), 0.))
    # Holds sums of G value for every grid position, used to assign a value to q
    g_sums = np.full((robot.grid.n_rows, robot.grid.n_cols, len(ACTIONS)), 0.)
    # Policy values
    policy = np.full((robot.grid.n_rows, robot.grid.n_cols, len(ACTIONS)),
                     1. / len(ACTIONS))

    # generate episodes until reach the max number of episodes
    for _ in tqdm(range(max_episodes)):
        # gradually reduce the epsilon parameter cause we need less exploration and more exploitation as the
        # episodes increase
        epsilon *= 0.99
        single_episode = generate_episodes(policy, robot)
        G = 0
        for idx, step in reversed(list(enumerate(single_episode[::-1]))):
            G = g * G + step[2]

            step_y = step[0][0]
            step_x = step[0][1]
            episode_action_idx = step[1]

            if (step[0], episode_action_idx) not in np.array(single_episode)[:, :2][idx + 1:]:

                # update returns(state,action) & q_grid(state,action)
                g_sums[step_y, step_x, episode_action_idx] += G
                q_grid[step_y, step_x, episode_action_idx] = g_sums[step_x, step_x, episode_action_idx] / idx

                # calculate the best action
                best_action_idx = -1
                max_value = -100000

                for action_idx in range(len(ACTIONS)):
                    if q_grid[step_y, step_x, action_idx] > max_value:
                        best_action_idx = action_idx
                        max_value = q_grid[step_y, step_x, action_idx]

                # update the policy matrix of the specific (state,action) pair
                # based on e-soft policy
                for action_idx in range(len(ACTIONS)):  # enumerate action space
                    # if state == step[0]:
                    if action_idx == best_action_idx:
                        policy[step_y, step_x, action_idx] = 1 - epsilon + epsilon / len(ACTIONS)
                    else:
                        policy[step_y, step_x, action_idx] = epsilon / len(ACTIONS)

    # when the episodes iteration finishes
    # choose the corresponding robot action based on the updated policy probabilities
    actions = []
    probabilities = []
    current_state = robot.pos
    for action in possible_actions[current_state]:
        actions.append(action)
        probabilities.append(policy[current_state, action])
    corresponding_action = random.choices(actions, weights=probabilities, k=1)[0]

    current_direction = robot.dirs[robot.orientation]

    while current_direction != corresponding_action:
        robot.rotate('r')
        current_direction = robot.dirs[robot.orientation]

    robot.move()
