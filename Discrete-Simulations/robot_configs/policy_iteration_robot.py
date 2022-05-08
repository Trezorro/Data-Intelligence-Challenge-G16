from typing import Union
import numpy as np
from environment import Robot
import logging

# Logging settings
logging.basicConfig(level=logging.INFO, force=True)

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING) # change to INFO or DEBUG for more detailed output

MATERIALS = {0: 'cell_clean', -1: 'cell_wall', -2: 'cell_obstacle', -3: 'cell_robot_n', -4: 'cell_robot_e',
             -5: 'cell_robot_s', -6: 'cell_robot_w', 1: 'cell_dirty', 2: 'cell_goal', 3: 'cell_death'}


REWARD_MAP = {
    -3: -1,   #  A robot position (so clean)    
    -2: -10,  #  Obstacle (gray)  
    -1: -10,  #  Wall (red)       
    0: -1,   #  Clean (green)    
    1: 2,   #  Dirty (white)    
    2: 1,   #  Goal (orange)    
    3: -50,   #  Death (red cross)
}

def robot_epoch(robot: Robot, gamma=0.2, min_delta=0.1):
    """Policy iteration epoch."""
    values = np.zeros_like(robot.grid.cells)
    robot.debug_values = np.zeros_like(robot.grid.cells) # these are visualized in the GUI
    robot.show_debug_values = True
    policy = np.ones((robot.grid.n_cols, robot.grid.n_rows), dtype=int)
    DIRECTIONS = list(robot.dirs.keys())
    MOVES = list(robot.dirs.values())

    logger.info("Starting policy iteration...")
    for iteration in range(200):
        # =================
        # policy evaluation
        # =================

        c = 1

        while True:
            delta = 0

            # iterate over states:
            for col in range(robot.grid.n_cols):
                for row in range(robot.grid.n_rows):
                    position = (col, row)
                    old_value = values[col, row]

                    value = 0
                    policy_move = MOVES[policy[col, row]]

                    for move in MOVES:

                        # Get square that bot would land on after move.
                        target_state = tuple(np.array(position) + np.array(move))

                        if not ((0 <= target_state[0] < robot.grid.n_cols) and
                                (0 <= target_state[1] < robot.grid.n_rows)):
                            continue

                        reward = reward_function(robot.grid.cells.item(target_state))
                        target_value = clipper(values[target_state]) # clipping is used to prevent overflow
                        coefficient = clipper((reward + gamma * target_value))
                        if move == policy_move:
                            # NOTE: The probabilities here are not correctly scaled, but should yield 
                            # similar results in the final comparison, using the argmax.
                            value += (1 - robot.p_move) * coefficient
                            value = clipper(value)
                        else:
                            value += robot.p_move * coefficient
                            value = clipper(value)

                    values[col, row] = value  # update value
                    delta = max(delta, abs(old_value - value))

            # Break if converged
            if delta < min_delta:
                logger.info(f"[Policy evaluation] converged in {c} iterations with {delta=}.")
                break

        # ==================
        # policy improvement
        # ==================

        # iterate over states:
        changed_policies = 0  # tracked to detect convergence
        for col in range(robot.grid.n_cols):
            for row in range(robot.grid.n_rows):
                position = (col, row) # the state we evaluate now
                old_policy = policy[position]
                q_values = []
                for move in MOVES:
                    # Get square that bot would land on after move.
                    target_state = tuple(np.array(position) + np.array(move))

                    if not ((0 <= target_state[0] < robot.grid.n_cols) and
                            (0 <= target_state[1] < robot.grid.n_rows)):
                        # If this move would take the bot off the grid, skip it.
                        q_values.append(np.NINF)
                        continue

                    reward = reward_function(robot.grid.cells[target_state])
                    target_value = clipper(values[target_state])
                    
                    # Here we omit the summation over the possible s' states, given action a,
                    # since the only varying factor is the value of the state that the action
                    # actually wants to go to. In the argmax, ony this factor matters, and we 
                    # can forget about the influence of the environment randomness.
                    q_value = reward + gamma * target_value
                    q_values.append(q_value)
                    robot.debug_values[col, row] = q_value  # visualized in the GUI
                new_policy = np.argmax(q_values)
                policy[position] = new_policy

                if new_policy != old_policy:
                    changed_policies += 1
        logger.info(f"[Policy improvement] {changed_policies} policies changed.")

        if changed_policies == 0:
            break

    logger.info("-----------------------------------------------------")
    logger.info(f"Policy iteration complete. Took {iteration} iterations.")

    target_orientation = DIRECTIONS[policy[robot.pos]]

    # If we don't have the wanted orientation, rotate clockwise until we do:
    while target_orientation != robot.orientation:
        robot.rotate('r')
    logger.info(f"LET'S MOVE!\n")

    if not robot.move():
        logger.warning("We hit a wall! Dummy!")


def reward_function(square_label: Union[int, float]) -> int:
    square_label = int(square_label)

    if "cell_robot" in MATERIALS[square_label]:  # if the square is one of the robots
        square_label = -3

    return REWARD_MAP[square_label]


def clipper(value: float):
    """Clips value between a range of -1M to +1M."""
    return max(-1000000., min(value, 1000000.))
