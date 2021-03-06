from environment import RobotBase

MATERIALS = {0: 'cell_clean', -1: 'cell_wall', -2: 'cell_obstacle', -3: 'cell_robot_e', -4: 'cell_robot_s',
             -5: 'cell_robot_w', -6: 'cell_robot_n', 1: 'cell_dirty', 2: 'cell_goal', 3: 'cell_death'}

REWARD_MAP = {
    -3: 0,    # A robot position (so clean)
    -2: -10,  # Obstacle (gray)
    -1: -10,  # Wall (red)
    0: 0,     # Clean (green)
    1: 4,     # Dirty (white)
    2: 1,     # Goal (orange)
    3: -50,   # Death (red cross)
}


def get_label_based_reward(square_label: int) -> int:
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


def get_label_and_battery_based_reward(square_label: int, battery_drained: bool) -> int:
    """ Function returns reward based on the label of a square and whether the battery was drained.

    Returns:
        Reward associated to the label and whether the battery was drained.

    Raises:
        ValueError: Parameter out of known range of labels.
    """
    if square_label < -6 or square_label > 3:
        raise ValueError("label_based_reward.get_reward: parameter out of range with value ", square_label)

    if "cell_robot" in MATERIALS[square_label]:
        square_label = -3

    reward = REWARD_MAP[square_label]

    if battery_drained:
        reward -= 1

    return reward


def get_reward_turning_off():
    """ Function returns reward for turning off the robot """
    return -10
