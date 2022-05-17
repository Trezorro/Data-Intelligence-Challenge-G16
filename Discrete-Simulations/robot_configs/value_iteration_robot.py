from environment import RobotBase
import numpy as np
import random

from helpers.reward_functions import get_label_based_reward


def robot_epoch(robot: RobotBase, gamma=0.9, theta=0.01):
    """ Value iteration bot """

    # Initialize parameters
    THETA = theta
    GAMMA = gamma
    MAX_ITERATIONS = 10000

    # Initialize V with all values to -1000 which are obstacles/walls
    V = np.zeros_like(robot.grid.cells)
    for x in range(robot.grid.n_cols):
        for y in range(robot.grid.n_rows):
            if -3 < robot.grid.cells[y][x] < 0 or robot.grid.cells[y][x] == 3:
                V[(y, x)] = -100

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
                if V[(y, x)] == -100:
                    continue

                # Store current value
                v = V[y, x]

                # Get rewards for each move
                rewards = {}
                for move in moves:
                    # Calculate the new position the robot would be in after the move
                    new_pos = tuple(np.array([y, x]) + move)

                    # If that would be an obstacle, the robot would not move
                    # This would in no way contribute to the goal, so it would not make sense to do this move.
                    # Therefore, it also makes no sense to let it contribute to the calculation of the states value.
                    if -3 < robot.grid.cells[new_pos] < 0:
                        continue

                    # Get the reward of the new square.
                    reward = get_label_based_reward(robot.grid.cells[new_pos])
                    rewards[move] = reward + GAMMA * V[new_pos]

                # Calculate the number of viable moves for the robot
                count_possible_moves = len(rewards)

                # Get the max new value of the state. For this, loop over all possible logical moves
                expected_values = []
                for move, reward in rewards.items():
                    expected_value = (1 - p_move) * reward

                    if p_move != 0:
                        for rand_move in rewards.keys():
                            expected_value += p_move / count_possible_moves * rewards[rand_move]

                    expected_values.append(expected_value)

                # Store new state value
                V_new[(y, x)] = max(expected_values)

                # Store new delta
                DELTA = max(DELTA, abs(v - V_new[(y, x)]))

        V = V_new

        # Compare delta to theta
        if DELTA < THETA:
            break

    robot_position = robot.pos

    # For debugging purposes (visualizing V_T gives the normal view of the field)
    V_T = V.T

    highest_val = -100000

    # Create a dict to save the value of each move
    corresponding_move = dict(zip(moves, [None] * len(moves)))
    top_moves = []
    for move in moves:
        new_pos = tuple(np.array(robot_position) + move)

        # If that would be an obstacle, skip, as doing that move would not contribute to the goal at all.
        if -3 < robot.grid.cells[new_pos] < 0:
            continue

        val = V[new_pos]

        corresponding_move[move] = val

        if val > highest_val:
            highest_val = val

    for move in corresponding_move.keys():
        if highest_val == corresponding_move[move]:
            top_moves.append(move)

    # Choose a random move from the best ones
    corresponding_move = random.choice(top_moves)

    current_direction = robot.dirs[robot.orientation]

    while current_direction != corresponding_move:
        robot.rotate('r')
        current_direction = robot.dirs[robot.orientation]

    robot.move()
