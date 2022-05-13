from typing import Union
import numpy as np
from environment import Robot
import logging

# Logging settings
from helpers.reward_functions import get_label_based_reward

logging.basicConfig(level=logging.INFO, force=True)

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING) # change to INFO or DEBUG for more detailed output

def robot_epoch(robot: Robot, gamma=0.2, min_delta=0.1):
    """Policy iteration epoch."""
    values = np.zeros_like(robot.grid.cells)
    robot.debug_values = np.zeros_like(robot.grid.cells) # these are visualized in the GUI
    robot.show_debug_values = True
    policy = np.ones_like(robot.grid.cells, dtype=int)
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
                    position = (row, col)
                    old_value = values[position]

                    value = 0
                    policy_move = MOVES[policy[position]]

                    for move in MOVES:

                        # Get square that bot would land on after move.
                        target_state = tuple(np.array(position) + np.array(move))

                        if not ((0 <= target_state[0] < robot.grid.n_rows) and
                                (0 <= target_state[1] < robot.grid.n_cols)):
                            continue

                        reward = get_label_based_reward(robot.grid.cells.item(target_state))
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

                    values[position] = value  # update value
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
                position = (row, col) # the state we evaluate now
                old_policy = policy[position]
                q_values = []
                for move in MOVES:
                    # Get square that bot would land on after move.
                    target_state = tuple(np.array(position) + np.array(move))

                    if not ((0 <= target_state[0] < robot.grid.n_rows) and
                            (0 <= target_state[1] < robot.grid.n_cols)):
                        # If this move would take the bot off the grid, skip it.
                        q_values.append(np.NINF)
                        continue

                    reward = get_label_based_reward(robot.grid.cells[target_state])
                    target_value = clipper(values[target_state])
                    
                    # Here we omit the summation over the possible s' states, given action a,
                    # since the only varying factor is the value of the state that the action
                    # actually wants to go to. In the argmax, ony this factor matters, and we 
                    # can forget about the influence of the environment randomness.
                    q_value = reward + gamma * target_value
                    q_values.append(q_value)
                    robot.debug_values[position] = q_value  # visualized in the GUI
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

    if not robot.move()[0]:
        logger.warning("We hit a wall! Dummy!")


def clipper(value: float):
    """Clips value between a range of -1M to +1M."""
    return max(-1000000., min(value, 1000000.))
