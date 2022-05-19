import copy
import random

import numpy as np

from environment import Robot
from helpers.label_based_reward import get_reward
from tqdm import tqdm


ACTIONS = ("n", "e", "s", "w", "off")


def generate_episodes(policy: np.ndarray, robot: Robot):
    """ Generate the episodes based on the policy-action probabilities.

    Args:
        policy: The policy array to be used to generate episodes to train on.
        robot: The robot object to be used.

    Returns a list called episodes that includes the state, the action and the reward of the next action i.e.
     [[(1, 1), (1, 0), 2], [(2, 1), (1, 0), -1], [(3, 1), (-1, 0), -1]]: the state (1,1) has next action (1,0)
     that gives 2 as reward, then, the state (2,1) has (1,0) as next action that gives -1 as reward etc.
    """

    # create a deepcopy of robot object because in the simulation
    # copy the current robot
    temp_robot = copy.deepcopy(robot)

    episodes = []
    position = temp_robot.pos

    step = 0
    # step of the episodes to be proportional to the cleaned tiles
    while step < 800:
        # choose randomly a action but based on action weights
        chosen_action = random.choices(ACTIONS,
                                       weights=policy[position[0], position[1]], k=1)[0]

        # update the reward in the simulated grid
        new_pos = tuple(np.asarray(position) + robot.dirs[chosen_action])
        reward = get_reward(temp_robot.grid.cells[new_pos])

        episodes.append([position, chosen_action, reward])

        step += 1

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
    q_grid = np.array((robot.grid.n_rows, robot.grid.n_cols, len(actions)))
    # Holds sums of G value for every grid position, used to assign a value to q
    g_sums = np.array((robot.grid.n_rows, robot.grid.n_cols, len(actions)))
    # Policy values
    policy = np.array((robot.grid.n_rows, robot.grid.n_cols, len(actions)))
    possible_actions = {}

    for x in range(robot.grid.n_cols):
        for y in range(robot.grid.n_rows):

            # ignore cells that are walls or obstacles
            if -3 < robot.grid.cells[y][x] < 0 or robot.grid.cells[y][x] == 3:
                continue

            # create a list with possible tiles

            # create a list with all the possible actions of each state
            possible_actions[(y, x)] = []

            for action in actions:
                # calculate the new position the robot would be in after the action
                new_pos = tuple(np.array([y, x]) + action)

                # ignore tiles that are either death or walls or obstacles
                if -3 < robot.grid.cells[new_pos] < 0 or robot.grid.cells[new_pos] == 3:
                    continue

                # keep track of the possible actions of a state
                possible_actions[(y, x)].append(action)

                # initialize returns and Q grid with empty list and 0 respectively for every state action combination
                q_grid[(y, x), action] = 0
                policy[(y, x), action] = 1 / len(possible_actions)

    # generate episodes until reach the max number of episodes
    for _ in tqdm(range(max_episodes)):
        # gradually reduce the epsilon parameter cause we need less exploration and more exploitation as the
        # episodes increase
        epsilon *= 0.99
        single_episode = generate_episodes(policy, robot, possible_actions)
        G = 0
        for idx, step in enumerate(single_episode[::-1]):
            G = g * G + step[2]
            # first-visit
            if (step[0], step[1]) not in np.array(single_episode[::-1])[:, :2][idx + 1:]:

                # update returns(state,action) & q_grid(state,action)
                returns[step[0], step[1]].append(G)
                q_grid[step[0], step[1]] = np.mean(returns[step[0], step[1]])

                # calculate the best action
                best_action = (0, 0)
                max_value = -100

                count_actions = 0
                # q_grid_lst = list(q_grid)
                for action in possible_actions[step[0]]:
                    # if state == step[0]:
                    count_actions += 1
                    if q_grid[step[0], action] > max_value:
                        best_action = action
                        max_value = q_grid[step[0], action]

                # update the policy matrix of the specific (state,action) pair
                # based on e-soft policy
                for action in possible_actions[step[0]]:  # enumerate action space
                    # if state == step[0]:
                    if action == best_action:
                        policy[step[0], action] = 1 - epsilon + epsilon / count_actions
                    else:
                        policy[step[0], action] = epsilon / count_actions

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
