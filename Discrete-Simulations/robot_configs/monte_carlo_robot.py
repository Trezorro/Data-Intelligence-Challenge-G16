from typing import Union
import numpy as np
from environment import Robot
import logging
import random

# Logging settings
from helpers.label_based_reward import get_reward


def robot_epoch(robot: Robot, gamma=0.2, min_delta=0.1):
    """ Initianlize the attributes needed for the Mote Carlo On Policy implementation

    :param max_episodes: the max number of episodes that we want to compute
    :param q_grid: the action state value, initialzied to 0. For every cell there are 4 possible moves so 4 times 0
    TODO not 4 possible moves in every state
    :param moves: list of all actions
    :param Returns: empty array for every possible combination of moves and states
    :param policy: initialize a policy for every possible state
    """
    max_episodes = 100
    max_steps_in_episodes = len(robot.grid.cells)
    q_grid = np.zeros((robot.grid.n_cols,robot.grid.n_rows,4))

    moves = list(robot.dirs.values())
    Returns = {}
    policy = {}
    for x in range(robot.grid.n_cols):
        for y in range(robot.grid.n_rows):
            if -3 < robot.grid.cells[y][x] < 0 or robot.grid.cells[y][x] == 3:
                continue
            possible_moves = []
            for move in moves:
                # Calculate the new position the robot would be in after the move
                new_pos = tuple(np.array([y, x]) + move)

                if robot.grid.cells[new_pos] < 0 or robot.grid.cells[new_pos] == 3:
                    continue
                possible_moves.append(move)
                Returns[str((y, x)) + ", " + str(move)] = []
            if len(possible_moves) > 1:
                weights = [0.1/len(possible_moves)]*len(possible_moves)
                policy[(y, x)] = random.choices(possible_moves,weights=weights,k=1)[0]
            elif len(possible_moves) == 1:
                policy[(y, x)] = possible_moves[0]

    """Generate episodes"""
    def generate_episodes(policy):
        """ Generate the episodes based on the current policy\
        :param position: the mosition of the robot after applying a move based on policy. Initialized as the
        current position
        :param policy: the current policy

        Returns the dict episodes that includes the state and the action generated for this episode
        """
        position = tuple(np.array(robot.pos))
        episodes = {}
        episodes[position] = policy[position]

        for i in range(max_steps_in_episodes - 1):
            new_pos = tuple(np.asarray(position)+ policy[position])
            episodes[new_pos] = policy[new_pos]
            position = new_pos

        return episodes



    """Implementation"""
    for episode in range(max_episodes):
        pass
