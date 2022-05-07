from environment import Robot, Grid
import numpy as np
import random


def robot_epoch(robot: Robot):
    """ Value iteration bot """
    print("Start new epoch")

    # Initialize parameters
    THETA = 0.001
    GAMMA = 0.9
    MAX_ITERATIONS = 10000

    # Initialize V with all values to -1000 which are obstacles/walls
    V = np.zeros_like(robot.grid.cells)
    for x in range(robot.grid.n_cols):
        for y in range(robot.grid.n_rows):
            if -3 < robot.grid.cells[x][y] < 0 or robot.grid.cells[x][y] == 3:
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

                    # If that would be an obstacle, the robot would not move
                    # This would in no way contribute to the goal, so it would not make sense to do this move.
                    # Therefore, it also makes no sense to let it contribute to the calculation of the states value.
                    if -3 < robot.grid.cells[new_pos] < 0:
                        continue

                    # Get the reward of the new square.
                    reward = get_reward(robot.grid, new_pos, robot)
                    rewards[move] = reward + GAMMA * V[new_pos]

                count_possible_moves = len(rewards)

                # Get the max new value of the state
                max_new_val = -10000
                for move, reward in rewards.items():
                    sum = 0
                    sum += (1 - p_move) * reward

                    if p_move != 0:
                        for rand_move in rewards.keys():
                            sum += p_move/count_possible_moves * rewards[rand_move]

                    if sum > max_new_val:
                        max_new_val = sum

                # Store new state value
                V_new[(x, y)] = max_new_val

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

    # Create a dict to save the value of each move
    corresponding_move = dict(zip(moves, [None] * len(moves)))
    top_moves = []
    for move in moves:
        new_pos = tuple(np.array(robot_position) + move)

        # If that would be an obstacle, skip, as it would not be useful
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

    print("Want to move to ", corresponding_move)

    current_direction = robot.dirs[robot.orientation]
    print(current_direction, corresponding_move)

    while current_direction != corresponding_move:
        robot.rotate('r')
        current_direction = robot.dirs[robot.orientation]

    robot.move()


def get_reward(grid: Grid, square, robot: Robot) -> float:
    reward_per_cell = 0

    if grid.cells[square] == 3:  # Death state is negative reward
        reward_per_cell += -200
    elif grid.cells[square] == 2:  # Goal state positive reward
        reward_per_cell += 3
    elif grid.cells[square] == 1:  # Dirty square positive reward
        reward_per_cell += 5
    elif grid.cells[square] == 0:  # Clear square very negative reward
        reward_per_cell += -2
    elif grid.cells[square] < -2:  # Robot square
        reward_per_cell += -2
    else:                           # Obstacles, negative reward
        reward_per_cell += -1

    return reward_per_cell
