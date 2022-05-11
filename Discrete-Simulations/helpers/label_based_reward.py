MATERIALS = {0: 'cell_clean', -1: 'cell_wall', -2: 'cell_obstacle', -3: 'cell_robot_n', -4: 'cell_robot_e',
             -5: 'cell_robot_s', -6: 'cell_robot_w', 1: 'cell_dirty', 2: 'cell_goal', 3: 'cell_death'}

REWARD_MAP = {
    -3: -1,  # A robot position (so clean)
    -2: -1,  # Obstacle (gray)
    -1: -1,  # Wall (red)
    0: -1,   # Clean (green)
    1: 2,    # Dirty (white)
    2: 1,    # Goal (orange)
    3: -666  # Death (red cross)
}


def get_reward(square_label: int) -> int:
    if "cell_robot" in MATERIALS[square_label]:
        square_label = -3

    return REWARD_MAP[square_label]