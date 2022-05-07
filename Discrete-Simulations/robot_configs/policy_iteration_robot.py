from typing import Union
import numpy as np
from environment import Robot
import logging
logging.basicConfig(level=logging.INFO, force=True)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
print("Loaded policy iteration robot.")

MATERIALS = {0: 'cell_clean', -1: 'cell_wall', -2: 'cell_obstacle', -3: 'cell_robot_n', -4: 'cell_robot_e',
                 -5: 'cell_robot_s', -6: 'cell_robot_w', 1: 'cell_dirty', 2: 'cell_goal', 3: 'cell_death'}

def reward_function(square_label: Union[int, float]) -> int:
    square_label = int(square_label)
    if "cell_robot" in MATERIALS[square_label]: # if the square is one of the robots
        square_label = -3
    REWARD_MAP = {
        -3: -1,   #  A robot position (so clean)    
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
    gamma = .9
    DIRECTIONS = list(robot.dirs.keys())
    MOVES = list(robot.dirs.values())
    MIN_DELTA = .001
    logger.info("Starting policy iteration...")

    # Policy iteration
    for iteration in range(10000):
        # policy evaluation
        c = 1
        while True:
            delta = 0
            # iterate over states:
            for col in range(robot.grid.n_cols):
                for row in range(robot.grid.n_rows):
                    position = (col, row)
                    
                    if robot.grid.cells.item(position) in (-1, -2):  # wall or obstacle
                        continue
                    old_value = values[col, row]
                    value = 0
                    policy_move = MOVES[policy[col, row]]
                    for move in MOVES:
                        # Get square that bot would land on after move.
                        target_state = tuple(np.array(position) + np.array(move))
                        reward = reward_function(robot.grid.cells.item(target_state))
                        if move == policy_move:
                            value += (1 - robot.p_move) * (reward + gamma * values[target_state])
                        else:
                            value += robot.p_move * (reward + gamma * values[target_state])
                            
                    values[col, row] = value  # update value
                    delta = max(delta, abs(old_value - value))
            if delta < MIN_DELTA:
                logger.info(f"[Policy evaluation] converged in {c} iterations with {delta=}.")
                break
            else:
                c+=1
                if c % 500 == 0:
                    logger.info(f"[Policy evaluation] at iteration {c} with {delta=}.")
        # policy improvement
        # iterate over states:
        changed_policies = 0
        for col in range(robot.grid.n_cols):
            for row in range(robot.grid.n_rows):
                position = (col, row)
                if robot.grid.cells.item(position) in (-1, -2):  # wall or obstacle
                        continue
                old_policy = policy[position]
                expected_values = []
                for move in MOVES:
                    # Get square that bot would land on after move.
                    target_state = tuple(np.array(position) + np.array(move))
                    reward = reward_function(robot.grid.cells[target_state])
                    expected_values.append(reward + gamma * values[target_state])
                new_policy = np.argmax(expected_values)
                policy[position] = new_policy
                if new_policy != old_policy:
                    changed_policies += 1
        logger.info(f"[Policy improvement] {changed_policies} policies changed.")
        if changed_policies == 0:
            break
    logger.info("-----------------------------------------------------")
    logger.info(f"Policy iteration complete. Took {iteration} iterations.")
    target_orientation = DIRECTIONS[policy[robot.pos]]
    while target_orientation != robot.orientation:
        # If we don't have the wanted orientation, rotate clockwise until we do:
        robot.rotate('r')
    logger.info(f"LET'S MOVE!\n")
    if not robot.move():
        logger.warn("We hit a wall! Dummy!")
    with np.printoptions(linewidth=150, precision=3):
        print(*values.T, sep='\n')
        