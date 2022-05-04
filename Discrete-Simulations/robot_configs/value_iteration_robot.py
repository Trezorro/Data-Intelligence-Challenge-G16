from typing import Tuple

from environment import Robot, Grid
import numpy as np


"""
PROBLEMS
Currently, the bot seems to calculate the value of each state.
However, once the bot moves, it does not see the difference between a square that has already been cleaned, and one that 
is dirty, even though a 'clean square' is heavily discouraged in the reward function.
"""

def robot_epoch(robot: Robot):
    """ Value iteration bot """
    print("Start new epoch")

    # Initialize parameters
    THETA = 0.01
    GAMMA = 0.6
    MAX_ITERATIONS = 200

    # Initialize V with all values to -1000 which are obstacles/walls
    V = np.zeros_like(robot.grid.cells)
    for x in range(robot.grid.n_cols):
        for y in range(robot.grid.n_rows):
            if -3 < robot.grid.cells[x][y] < 0:
                V[(x,y)] = -1000

    moves = list(robot.dirs.values())
    p_move = robot.p_move

    # For max iterations
    for iter in range(MAX_ITERATIONS):
        DELTA = 0

        # Loop over all states
        for x in range(robot.grid.n_cols):
            for y in range(robot.grid.n_rows):
                # If obstacle, skip iteration
                if V[(x,y)] == -1000:
                    continue

                # Store current value
                v = V[x, y]

                # Get rewards for each move
                rewards = {}
                for move in moves:
                    new_pos = tuple(np.array([x, y]) + move)
                    reward = get_reward(robot.grid, new_pos, robot)
                    rewards[move] = reward + GAMMA * V[new_pos]

                # Calculate aggregate expected reward for doing a random move
                random_reward_sum = 0
                for move in moves:
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
                V[(x,y)] = max_new_val

                # Store new delta
                DELTA = max(DELTA, abs(v - max_new_val))

        # Compare delta to theta. Currently commented out for lots of convergence :)
        # if DELTA < THETA:
        #     break

    robot_position = robot.pos

    # For debugging purposes (visualizing V_T gives the normal view of the field)
    V_T = V.T

    highest_val = -100000
    corresponding_move = None

    for move in moves:
        new_pos = tuple(np.array(robot_position) + move)
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
        reward_per_cell += -100
    elif grid.cells[square] == 2:  # Goal state positive reward
        reward_per_cell += 100
    elif grid.cells[square] == 1:  # Dirty square positive reward
        reward_per_cell += 200
    elif grid.cells[square] == 0:  # Clear square very negative reward
        reward_per_cell += -1000
    elif grid.cells[square] < -2:  # Robot square neutral
        reward_per_cell += 1000
    else:                           # Obstacles, negative reward
        reward_per_cell += -100

    return reward_per_cell
