from turtle import position
import numpy as np
from environment import Robot


def reward_function(square_label: int):
    REWARD_MAP = {
        -2: -1,  #  Obstacle (gray)  
        -1: -1,  #  Wall (red)       
        0: -1,   #  Clean (green)    
        1: 2,   #  Dirty (white)    
        2: 1,   #  Goal (orange)    
        3: -666,   #  Death (red cross)
    }
    return REWARD_MAP[square_label]

def robot_epoch(robot: Robot):
    # figure out the policy
    # initialize values and policy
    values = np.zeros_like(robot.grid.cells)
    policy = np.ones((robot.grid.n_cols, robot.grid.n_rows),
                     dtype=int)
    gamma = 0.9
    DIRECTIONS = list(robot.dirs.keys())
    MOVES = list(robot.dirs.values())
    MIN_DELTA = .01

    # Policy iteration
    while True:  
        # policy evaluation
        delta = 0
        while True:
            # iterate over states:
            for col in range(robot.grid.n_cols):
                for row in range(robot.grid.n_rows):
                    position = np.array([col, row])
                    
                    if robot.grid.cells[col, row] == 0:
                        continue
                    old_value = values[col, row]
                    value = 0
                    policy_move = MOVES[policy[col, row]]
                    for move in MOVES:
                        # Get square that bot would land on after move.
                        target_state = tuple(position + np.array(move))
                        reward = reward_function(robot.grid.cells[target_state])
                        if move == policy_move:
                            value += (1 - robot.p_move) * (reward + gamma * values[target_state])
                        else:
                            value += robot.p_move * (reward + gamma * values[target_state])
                            
                    values[col, row] = value  # update value
                    delta = max(delta, abs(old_value - value))
            if delta < MIN_DELTA:
                break
        # policy improvement
        # iterate over states:
        policy_stable = True
        for col in range(robot.grid.n_cols):
            for row in range(robot.grid.n_rows):
                position = np.array([col, row])
                old_policy = policy[col, row]
                expected_values = []
                for move in MOVES:
                    # Get square that bot would land on after move.
                    target_state = tuple(position + np.array(move))
                    reward = reward_function(robot.grid.cells[target_state])
                    expected_values.append(reward + gamma * values[target_state])
                new_policy = np.argmax(expected_values)
                policy[col, row] = new_policy
                if new_policy != old_policy:
                    policy_stable = False
        if policy_stable:
            target_orientation = DIRECTIONS[policy[robot.pos]]
            while target_orientation != robot.orientation:
            # If we don't have the wanted orientation, rotate clockwise until we do:
            # print('Rotating right once.')
                robot.rotate('r')
            robot.move()