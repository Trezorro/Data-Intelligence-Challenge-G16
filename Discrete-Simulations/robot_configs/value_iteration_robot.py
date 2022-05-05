from typing import Tuple

from environment import Robot, Grid
import numpy as np


"""
PROBLEMS
Current problems: it has no sense of 'time pressure' and doesn't seem to want to move... 

As all squares have the possibility to end up in a 'dirty square' in one move, they are all valued the same, 
it picks the first move, which is run into a wall. This is very visible with Gamma = 0.
"""

def robot_epoch(robot: Robot):
    """ Value iteration bot """
    print("Start new epoch")

    # Initialize parameters
    THETA = 0.1
    GAMMA = 0
    MAX_ITERATIONS = 40

    # Initialize V with all values to -1000 which are obstacles/walls
    V = np.zeros_like(robot.grid.cells)
    for x in range(robot.grid.n_cols):
        for y in range(robot.grid.n_rows):
            if -3 < robot.grid.cells[x][y] < 0:
                V[(x,y)] = -100

    moves = list(robot.dirs.values())
    p_move = robot.p_move

    # For max iterations
    for iter in range(MAX_ITERATIONS):
        DELTA = 0

        V_new = V.copy()

        # Loop over all states
        for x in range(robot.grid.n_cols):
            for y in range(robot.grid.n_rows):
                # If obstacle, skip iteration
                if V[(x,y)] == -100:
                    continue

                # Store current value
                v = V[x, y]

                # Get rewards for each move
                rewards = {}
                for move in moves:
                    # Calculate the new position the robot would be in after the move
                    new_pos = tuple(np.array([x, y]) + move)

                    # If that would be an obstacle, the robot would not move, so reset the position
                    if -3 < robot.grid.cells[new_pos] < 0:
                        new_pos = tuple(np.array([x,y]))

                    # Get the reward of the new square.
                    # TODO in goal states next to a wall, it would probably just stay still?
                    reward = get_reward(robot.grid, new_pos, robot)
                    rewards[move] = reward + GAMMA * V[new_pos]

                # Calculate aggregate expected reward for doing a random move
                random_reward_sum = 0
                for move in moves:
                    # The environment only chooses random moves that do something
                    if robot.grid.cells[tuple(np.array([x,y]) + move)] > 0:
                        random_reward_sum += 1 / 4 * rewards[move]

                # Get the max new value of the state
                max_new_val = -10000
                for move in moves:
                    sum = 0
                    sum += (1 - p_move) * rewards[move]
                    sum += (p_move) * random_reward_sum

                    if sum > max_new_val:
                        max_new_val = sum

                # Store new state value
                V_new[(x,y)] = max_new_val

                # Store new delta
                DELTA = max(DELTA, abs(v - max_new_val))

        V = V_new

        # Compare delta to theta
        if DELTA < THETA:
            print("breaking at ", iter, " iterations!")
            break

    robot_position = robot.pos

    # For debugging purposes (visualizing V_T gives the normal view of the field)
    V_T = V.T

    highest_val = -100000
    corresponding_move = None

    for move in moves:
        new_pos = tuple(np.array(robot_position) + move)

        # If that would be an obstacle, the robot would not move, so reset the position
        if -3 < robot.grid.cells[new_pos] < 0:
            new_pos = tuple(np.array(robot_position))

        val = V[new_pos]

        if val > highest_val:
            highest_val = val
            corresponding_move = move

    print("Want to move to ", corresponding_move)

    current_direction = robot.dirs[robot.orientation]
    print(current_direction, corresponding_move)

    while current_direction != corresponding_move:
        robot.rotate('r')
        current_direction = robot.dirs[robot.orientation]

    robot.move()


def get_reward(grid: Grid, square, robot: Robot) -> float:
    reward_per_cell = 0

    if grid.cells[square] == 3: # Death state is negative reward
        reward_per_cell += -2
    elif grid.cells[square] == 2:  # Goal state positive reward
        reward_per_cell += 1
    elif grid.cells[square] == 1:  # Dirty square positive reward
        reward_per_cell += 1
    elif grid.cells[square] == 0:  # Clear square very negative reward
        reward_per_cell += -2
    elif grid.cells[square] < -2:  # Robot square
        reward_per_cell += -1
    else:                           # Obstacles, negative reward
        reward_per_cell += -1

    return reward_per_cell
