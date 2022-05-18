from typing import Dict, Tuple


import numpy as np
import random


class RobotBase:
    def __init__(self, grid, pos, orientation, p_move=0, battery_drain_p=1,
                 battery_drain_lam=0, vision=1):
        """Base class for all robots.

        Args:
            grid: The grid the robot will move on.
            pos: Current position as a tuple (y, x).
            orientation: Current orientation of the robot. One of "n", "e", "s",
                "w".
            p_move: Probability of robot performing a random move instead of
                listening to a given command.
            battery_drain_p: Probability of a battery drain event happening at
                each move.
            battery_drain_lam: Amount (lambda) of battery drain (X) in
                X ~ Exponential(lambda)
            vision: Number of tiles in each of the 4 directions included in the
                robots vision.
        """
        if grid.cells[pos[0], pos[1]] != 1:
            raise ValueError

        # Current orientation of the robot, one of 'n', 'e', 's', 'w'.
        self.orientation: str = orientation
        # Grid associated numbers of various robot orientations
        self.orients = {'e': -3, 's': -4, 'w': -5, 'n': -6}
        self.dirs = {'e': (0, 1), 's': (1, 0), 'w': (0, -1), 'n': (-1, 0)}
        self.dirs_inv = {(0, 1): 'e', (1, 0): 's', (0, -1): 'w', (-1, 0): 'n'}

        self.pos: Tuple[int, int] = pos
        self.grid: Grid = grid
        self.p_move = p_move
        self.battery_drain_p = battery_drain_p
        self.battery_drain_lam = battery_drain_lam
        self.vision = vision

        self.grid.cells[pos] = self.orients[self.orientation]
        # historic x and y coordinates of the robot [[y1, y2,...], [x1, x2,...]]
        self.history = [[pos[0]], [pos[1]]]
        self.battery_lvl = 100  # Current battery level (in %)
        self.alive = True  # Indicator of whether the robot is alive

        # Array to store debug values for each tile
        self.debug_values = np.zeros_like(grid.cells)
        self.show_debug_values = False

    def possible_tiles_after_move(self) -> Dict[Tuple[int, int], int]:
        """Returns the values of squares the robot can see from its current
        position.

        Returns:
            Dictionary containing the moves relative to the robots current
            position that the robot can do (as keys) and the value of the
            resulting square (values)

            example:
            {
            ( 0,-1):  1,    Dirty square
            ( 1, 0):  0,    Clean square
            ( 0, 1): -1,    Wall (you will not move if you do this action)
            (-1, 0): -2,    Obstacle (you will not move if you do this action)
            ( 0,-2):  2,    Goal (it is not possible to move here in one go,
                                  it is 2 squares away)
            ( 2, 0):  0,         (it is not possible to move here in one go, it
                                  is 2 squares away)
            ( 0, 2):  0,         (it is not possible to move here in one go, it
                                  is 2 squares away)
            (-2, 0):  0          (it is not possible to move here in one go, it
                                  is 2 squares away)
        }
        """

        # Obtain possible movement directions
        moves = list(self.dirs.values())

        # Initialize dictionary to return
        data = {}

        # Loop over distance the robot can see
        for i in range(self.vision):

            # Loop over possible moves
            for move in moves:
                # Get square that bot would land on after move.
                to_check = (np.array(self.pos) + (np.array(move) * (i + 1)))

                # Check whether the square is inside the playing field.
                if (self.grid.cells.shape[0] > to_check[0] >= 0
                        and self.grid.cells.shape[1] > to_check[1] >= 0):
                    # Store move with corresponding grid value that would be
                    # obtained.
                    data[(np.array(move) * (i + 1))] = self.grid.cells[to_check]
        return data

    def has_visited(self, x: int, y: int) -> bool:
        """ Function returns whether bot has visited a given coordinate.

        Args:
            x: the respective x coordinate.
            y: the respective y coordinate.

        Returns:
            Whether the robot has visited the coordinate. True if it has.
        """
        for i in range(len(self.history[0])):
            if self.history[0][i] == y and self.history[1][i] == x:
                return True

        return False

    def move(self) -> Tuple[bool, bool]:
        """ Function that moves the robot.

        This function simulates the movement of the robot. It will try to move
        forwards in the current direction it is facing. The battery is drained
        with probability `self.battery_drain_p`. A move in a random direction is
        done with probability `self.p_move`.

        Returns:
            first boolean indicates
                True: the bot has moved successfully.
                False: the bot did not move or the bot moved and died.
            second boolean indicates whether the battery was drained during the
            move.
        """
        def move_and_remember_or_die(pos, orient):
            """Used in both branches of movement so refactored here. """
            # If the bot can move to that square, move the bot and adapt the
            # grid.
            tile_after_move = self.grid.cells[pos]
            self.grid.cells[self.pos] = 0
            self.grid.cells[new_pos] = self.orients[orient]
            self.pos = new_pos
            self.history[0].append(self.pos[0])
            self.history[1].append(self.pos[1])

            # If moved to death square, then the bot dies.
            if tile_after_move == 3:
                self.alive = False
                return False, do_battery_drain
            return True, do_battery_drain

        # Can't move if we're dead now, can we?
        if not self.alive:
            return False, False

        # Decide whether to do a random move.
        random_move = np.random.binomial(1, self.p_move)

        # Decide whether this move will drain the battery
        do_battery_drain = (np.random.binomial(1, self.battery_drain_p) == 1)

        # If battery should be drained, drain the battery according to
        # exponential drain
        if do_battery_drain and self.battery_lvl > 0:
            self.battery_lvl -= np.random.exponential(self.battery_drain_lam)

        # Handle empty battery --> die
        if self.battery_lvl <= 0:
            self.alive = False
            return False, do_battery_drain

        # If random move, execute a random move
        if random_move == 1:
            # Get possible moves, choose a random one and calculate the square
            # the robot would supposedly end up on.
            moves = self.possible_tiles_after_move()
            random_move = random.choice([m for m in moves if moves[m] >= 0])
            dirs_keys = list(self.dirs.keys())
            random_move_index = list(self.dirs.values()).index(random_move)
            new_pos = tuple(np.array(self.pos) + random_move)
            new_orient = dirs_keys[random_move_index]
        else:
            new_pos = (np.array(self.pos) + self.dirs[self.orientation])
            new_orient = self.orientation

        if self.grid.cells[new_pos] >= 0:
            return move_and_remember_or_die(new_pos, new_orient)

        # If we cannot move, just stand still and leave the bot
        else:
            return False, do_battery_drain

    def rotate(self, direction: str) -> None:
        """ Rotates the robot in a given direction either left 'l' or right 'r'.

        This function rotates the robot either to the left or right based on the
        given parameter value 'l' or 'r', respectively. The function adapts
        Robot object orientation attribute and the grid value that the robot is
        currently positioned on.
        """

        # # == Drain the battery
        # # Decide whether this move will drain the battery
        # do_battery_drain = np.random.binomial(1, self.battery_drain_p)
        #
        # # If battery should be drained, drain the battery according to
        # exponential drain
        # if do_battery_drain == 1 and self.battery_lvl > 0:
        #     self.battery_lvl -= np.random.exponential(self.battery_drain_lam)

        # == Rotate the bot
        # Get index of orientation from list ['n', 'e', 's', 'w']
        current = list(self.orients.keys()).index(self.orientation)

        # Get new orientation based on index of current orientation, rotating
        # direction and list of orientations
        if direction == 'r':
            self.orientation = list(self.orients.keys())[(current + 1) % 4]
        elif direction == 'l':
            self.orientation = list(self.orients.keys())[current - 1]

        # Adapt grid value based on new orientation of the robot.
        self.grid.cells[self.pos] = self.orients[self.orientation]


class Grid:
    def __init__(self, n_cols: int, n_rows: int):
        self.n_rows = n_rows
        self.n_cols = n_cols

        self.cells = np.ones((n_rows, n_cols))

        # Building the boundary of the grid:
        self.cells[0, :] = self.cells[-1, :] = -1
        self.cells[:, 0] = self.cells[:, -1] = -1
        self.transposed_version = True

    def get(self, x: int, y: int) -> int:
        """ Returns value of cell in field

        Args:
            x: the x coordinate of the grid cell requested.
            y: the y coordinate of the grid cell requested.

        Returns:
            The value of the requested cell.
        """
        return self.cells[y][x]

    def get_c(self, coord_y_x: Tuple[int, int]):
        """ Returns the value of the cell in field

        Args:
            coord_y_x: The coordinate (y, x) of the grid cell requested.

        Returns:
            The value of the requested cell.
        """
        return self.cells[coord_y_x]

    def put(self, x: int, y: int, value: int) -> None:
        """ Modifies the grid by putting in a new value

        Args:
            x:      the x coordinate of the grid cell to adapt.
            y:      the y coordinate of the grid cell to adapt.
            value:  the new value of the cell.
        """
        self.cells[y][x] = value

    def put_c(self, coord_y_x: Tuple[int, int], value: int) -> None:
        """ Modifies the grid by putting in a new value

        Args:
            coord_y_x:  The coordinate (y, x) of the grid cell to adapt.
            value:      the new value of the cell.
        """
        self.cells[coord_y_x] = value

    def is_cleaned(self) -> bool:
        """ Function returns whether there are dirty squares left in the grid.

        Returns:
            Whether there are dirty squares left in the grid.
        """
        return not (1 in self.cells)

    def put_obstacle(self, x0, x1, y0, y1, from_edge=1) -> None:
        """ Builds an obstacle on the grid starting on xy0 and ending at xy1
        """
        self.cells[
                max(y0, from_edge): min(y1 + 1, self.n_rows - from_edge),
                max(x0, from_edge): min(x1 + 1, self.n_cols - from_edge)
            ] = -2

    def put_singular_obstacle(self, x, y) -> None:
        """ Puts obstacle tile at provided (x,y) """
        self.put(x, y, -2)

    def put_singular_goal(self, x, y) -> None:
        """ Puts a goal tile at provided (x,y) """
        self.put(x, y, 2)

    def put_singular_death(self, x, y) -> None:
        """ Puts death tile at provided (x,y) """
        self.put(x, y, 3)


def generate_grid(n_cols, n_rows):
    # Placeholder function used to generate a grid.
    # Select an empty grid file in the user interface and add code her to
    # automatically fill it.
    # Look at grid_generator.py for inspiration.
    grid = Grid(n_cols, n_rows)
    return grid
