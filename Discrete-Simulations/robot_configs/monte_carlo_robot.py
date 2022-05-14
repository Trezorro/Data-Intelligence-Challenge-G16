from typing import Union
import numpy as np
from environment import Robot
import logging
import random

# Logging settings
from helpers.label_based_reward import get_reward


def robot_epoch(robot: Robot, gamma=0.2, max_episodes = 100, epsilon = 0.1):
    """ Initianlize the attributes needed for the Mote Carlo On Policy implementation

    :param max_episodes: the max number of episodes that we want to compute
    :param q_grid: the action state value, initialzied to 0. For every cell there are 4 possible moves so 4 times 0
    :param moves: list of all actions
    :param Returns: empty array for every possible combination of moves and states
    :param policy: initialize a policy for every possible state
    """
    max_episodes = max_episodes
    g = gamma
    max_steps_in_episodes = 0
    q_grid = {}
    epsilon = epsilon
    moves = list(robot.dirs.values())
    Returns = {}
    policy = {}

    for x in range(robot.grid.n_cols):
        for y in range(robot.grid.n_rows):
            if -3 < robot.grid.cells[y][x] < 0 or robot.grid.cells[y][x] == 3:
                continue
            max_steps_in_episodes += 1
            possible_moves = []
            for move in moves:
                # Calculate the new position the robot would be in after the move
                new_pos = tuple(np.array([y, x]) + move)

                if robot.grid.cells[new_pos] < 0 or robot.grid.cells[new_pos] == 3:
                    continue
                possible_moves.append(move)
                Returns[(y, x),move] = []
                q_grid[(y, x), move] = 0
            policy[(y, x)] = []
            for move in possible_moves:
                policy[(y, x), move] = 1 / len(possible_moves)


    """Generate episodes"""
    def generate_episodes(policy):
        """ Generate the episodes based on the current policy\
        :param position: the position of the robot after applying a move based on policy. Initialized as the
        current position
        :param policy: the current policy probabilties

        Returns a list called episodes that includes the state, the move and the reward of the next move i.e.
         [[(1, 1), (1, 0), 2], [(2, 1), (1, 0), -1], [(3, 1), (-1, 0), -1]]: the state (1,1) has next action (1,0)
         that gives 2 as reward, then, the state (2,1) has (1,0) as next action that gives -1 as reward etc.
        """
        episodes= []
        position = tuple(np.array(robot.pos))

        for i in range(max_steps_in_episodes - 1):
            moves = []
            probs = []
            for state, action in policy.keys():
                if state == position:
                    moves.append(action)
                    probs.append(policy[state,action]
            chosen_move = random.choices(moves, weights=probs, k=1)[0]
            episodes.append([position, chosen_move])
            new_pos = tuple(np.asarray(position)+ chosen_move)
            reward = get_reward(robot.grid.cells[new_pos])
            episodes[i].append(reward)

            position = new_pos

        return episodes

    print("Episodes", generate_episodes(policy))

    """Implementation"""
    for episode in range(max_episodes):
        # generate episodes
        single_episode = generate_episodes(policy)
        G = 0
        for idx, step in enumerate(single_episode[::-1]):
            G = g * G + step[2]
            # first-visit
            if step[0] not in np.array(single_episode[::-1])[:, 0][idx + 1:]:
                Returns[step[0], step[1]].append(G)
                q_grid[step[0], step[1]] = np.mean(Returns[step[0], step[1]])
                best_action = (0, 0)
                max_value = -100

                count_actions = 0
                for state, action in q_grid.keys():
                    if state == step[0]:
                        count_actions += 1
                        if q_grid[state, action] > max_value:
                            best_action = action
                            max_value = q_grid[state, action]

                for state, action in q_grid.keys():  # enumerate action space
                    if state == step[0]:
                        if action == best_action:
                            policy[state,action] = 1 - epsilon + epsilon / count_actions
                        else:
                            policy[state,action] =  epsilon / count_actions

    moves = []
    probs = []
    for state, action in policy.keys():
        if state == tuple(np.array(robot.pos)):
            moves.append(action)
            probs.append(policy[state, action])
    corresponding_move = random.choices(moves, weights=probs, k=1)[0]

    current_direction = robot.dirs[robot.orientation]

    while current_direction != corresponding_move:
        robot.rotate('r')
        current_direction = robot.dirs[robot.orientation]

    robot.move()