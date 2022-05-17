class TDState:
    def __init__(self, pos_x: int, pos_y: int, vision: dict):
        """State for TD Lookup.

        Vision dict should have keys "n", "e", "w", "s" for which the values
        are True if clean and False if dirty. Walls and obstacles are always
        True.
        """
        self.pos_x = pos_x
        self.pos_y = pos_y
        self.vision = vision

    def get_index(self, action):
        """Get index of Q table for TD given for this state.

        The Q table has 4 dimensions. The first 2 are the physical grid (indexed y,x), the 3rd
        dimension is the combination of all possible vision states (len = 16),
        and the 4th dimension is the possible actions.
        """
        y = self.pos_y
        x = self.pos_x
        z = self.vision["n"] * 1 \
            + self.vision["e"] * 2 \
            + self.vision["s"] * 4 \
            + self.vision["w"] * 8

        action_map = {"n": 0,
                      "e": 1,
                      "s": 2,
                      "w": 3}
        i = action_map[action] if action is not None else None

        return y, x, z, i

    def make_copy(self):
        return TDState(self.pos_x, self.pos_y, self.vision)
