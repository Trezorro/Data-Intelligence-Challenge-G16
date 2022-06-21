import random


def get_action_random_agent(_):
    """ Function generates a random move. This random move consists on one random float between
    -1 and 1, and one random int either being 0 or 1.

    returns:
        turn: float between -1 and 1
        move: int between 0 and 1
    """
    return random.random() * 2 - 1, random.randint(0, 1)
