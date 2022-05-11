MATERIALS = {0: 'cell_clean', -1: 'cell_wall', -2: 'cell_obstacle', -3: 'cell_robot_w', -4: 'cell_robot_s',
             -5: 'cell_robot_e', -6: 'cell_robot_n', 1: 'cell_dirty', 2: 'cell_goal', 3: 'cell_death'}

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
    """ Function returns reward based on the label of a square.

    Labels of the squares have to lie within a range of (-6, 3)

    Returns:
        Reward associated to the label, as defined in REWARD_MAP.

    Raises:
        ValueError: Parameter out of known range of labels.
    """
    if square_label < -6 or square_label > 3:
        raise ValueError("label_based_reward.get_reward: parameter out of range with value ", square_label)

    if "cell_robot" in MATERIALS[square_label]:
        square_label = -3

    return REWARD_MAP[square_label]