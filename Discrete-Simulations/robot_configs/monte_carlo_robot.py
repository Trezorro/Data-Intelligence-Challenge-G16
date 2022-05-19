import copy
import random

import numpy as np

from environment import Robot
from helpers.label_based_reward import get_reward


def generate_episodes(policy, robot: Robot, number_of_tiles, possible_actions):
    """ Generate the episodes based on the policy-action probabilities stored in policy dict

    Returns a list called episodes that includes the state, the action and the reward of the next action i.e.
     [[(1, 1), (1, 0), 2], [(2, 1), (1, 0), -1], [(3, 1), (-1, 0), -1]]: the state (1,1) has next action (1,0)
     that gives 2 as reward, then, the state (2,1) has (1,0) as next action that gives -1 as reward etc.
    """
    # create a deepcopy of robot object because in the simulation
    # copy the current robot
    temp_robot = copy.deepcopy(robot)

    # create a list to store the episodes
    episodes = []

    # position = random.choice(all_possible_tiles)
    # TODO Reaction this comment
    position = temp_robot.pos

    step = 0
    # step of the episodes to be proportional to the cleaned tiles
    while step < number_of_tiles / 2:
        actions = []
        probabilities = []
        for action in possible_actions[position]:
            # if state == position:
            actions.append(action)
            probabilities.append(policy[position, action])
        # choose randomly a action but based on action weights
        chosen_action = random.choices(actions, weights=probabilities, k=1)[0]
        episodes.append([position, chosen_action])
        new_pos = tuple(np.asarray(position) + chosen_action)

        # update the reward in the simulated grid
        reward = get_reward(temp_robot.grid.cells[new_pos])
        episodes[step].append(reward)

        step += 1

        # update position of the simulated robot
        position = new_pos
        temp_robot.pos = position

    return episodes


def robot_epoch(robot: Robot, g=0.99, max_episodes=10, epsilon=0.99):
    """ Initialize the attributes needed for the Monte Carlo On Policy implementation"""
    max_episodes = max_episodes

    # count the clean tiles
    count_clean = 0

    # number of non wall or obstacle tiles
    number_of_tiles = 0

    q_grid = {}
    epsilon = epsilon
    actions = list(robot.dirs.values())
    Returns = {}  # Returns(state, action)
    policy = {}  # policy(state, action)
    all_possible_tiles = []
    possible_actions = {}

    for x in range(robot.grid.n_cols):
        for y in range(robot.grid.n_rows):

            # ignore cells that are walls or obstacles
            if -3 < robot.grid.cells[y][x] < 0 or robot.grid.cells[y][x] == 3:
                continue

            # create a list with possible tiles
            all_possible_tiles.append(tuple(np.array([y, x])))

            # create a list with all the possible actions of each state
            possible_actions[(y, x)] = []

            # count the number of tiles
            number_of_tiles += 1

            # TODO reaction, keep for calibration
            # count number of clean tiles
            if robot.grid.cells[y][x] == 0:
                count_clean += 1

            for action in actions:
                # calculate the new position the robot would be in after the action
                new_pos = tuple(np.array([y, x]) + action)

                # ignore tiles that are either death or walls or obstacles
                if -3 < robot.grid.cells[new_pos] < 0 or robot.grid.cells[new_pos] == 3:
                    continue

                # keep track of the possible actions of a state
                possible_actions[(y, x)].append(action)

                # initialize Returns and Q grid with empty list and 0 respectively for every state action combination
                Returns[(y, x), action] = []
                q_grid[(y, x), action] = 0
                policy[(y, x), action] = 1 / len(possible_actions)

    """ Monte Carlo On Policy implementation"""
    if (number_of_tiles - count_clean) < 20:
        max_episodes *= 10
    else:
        max_episodes = max_episodes * (number_of_tiles - count_clean)

    # generate episodes until reach the max number of episodes
    for episode in range(max_episodes):
        # gradually reduce the epsilon parameter cause we need less exploration and more exploitation as the
        # episodes increase
        epsilon *= 0.99
        single_episode = generate_episodes(policy, robot, number_of_tiles, possible_actions)
        G = 0
        for idx, step in enumerate(single_episode[::-1]):
            G = g * G + step[2]
            # first-visit
            if (step[0], step[1]) not in np.array(single_episode[::-1])[:, :2][idx + 1:]:

                # update Returns(state,action) & q_grid(state,action)
                Returns[step[0], step[1]].append(G)
                q_grid[step[0], step[1]] = np.mean(Returns[step[0], step[1]])

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
