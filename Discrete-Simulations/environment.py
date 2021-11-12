import numpy as np
import random


class Robot:
    def __init__(self, grid, pos, orientation, p_move=0, battery_drain_p=0, battery_drain_lam=0, vision=1):
        if grid.cells[pos[0], pos[1]] != 1:
            raise ValueError
        self.orientation = orientation
        self.pos = pos
        self.grid = grid
        self.orients = {'n': -3, 'e': -4, 's': -5, 'w': -6}
        self.dirs = {'n': (0, -1), 'e': (1, 0), 's': (0, 1), 'w': (-1, 0)}
        self.grid.cells[pos] = self.orients[self.orientation]
        self.history = [[], []]
        self.p_move = p_move
        self.battery_drain_p = battery_drain_p
        self.battery_drain_lam = battery_drain_lam
        self.battery_lvl = 100
        self.alive = True
        self.vision = vision

    def possible_tiles_after_move(self):
        moves = list(self.dirs.values())
        # Fool the robot and show a death tile as normal (dirty)
        data = {}
        for i in range(self.vision):
            for move in moves:
                to_check = tuple(np.array(self.pos) + (np.array(move) * (i + 1)))
                if to_check[0] < self.grid.cells.shape[0] and to_check[1] < self.grid.cells.shape[1] and to_check[
                    0] >= 0 and to_check[1] >= 0:
                    data[tuple(np.array(move) * (i + 1))] = self.grid.cells[to_check]
                    # Show death tiles as dirty:
                    if data[tuple(np.array(move) * (i + 1))] == 3:
                        data[tuple(np.array(move) * (i + 1))] = 1
        return data

    def move(self):
        # Can't move if we're dead now, can we?
        if not self.alive:
            return False
        random_move = np.random.binomial(1, self.p_move)
        do_battery_drain = np.random.binomial(1, self.battery_drain_p)
        if do_battery_drain == 1 and self.battery_lvl > 0:
            self.battery_lvl -= np.random.exponential(self.battery_drain_lam)
        # Handle empty battery:
        if self.battery_lvl <= 0:
            self.alive = False
            return False
        if random_move == 1:
            moves = self.possible_tiles_after_move()
            random_move = random.choice([move for move in moves if moves[move] >= 0])
            new_pos = tuple(np.array(self.pos) + random_move)
            # Only move to non-blocked tiles:
            if self.grid.cells[new_pos] >= 0:
                new_orient = list(self.dirs.keys())[list(self.dirs.values()).index(random_move)]
                tile_after_move = self.grid.cells[new_pos]
                self.grid.cells[self.pos] = 0
                self.grid.cells[new_pos] = self.orients[new_orient]
                self.pos = new_pos
                self.history[0].append(self.pos[0])
                self.history[1].append(self.pos[1])
                if tile_after_move == 3:
                    self.alive = False
                    return False
                return True
            else:
                return False
        else:
            new_pos = tuple(np.array(self.pos) + self.dirs[self.orientation])
            # Only move to non-blocked tiles:
            if self.grid.cells[new_pos] >= 0:
                tile_after_move = self.grid.cells[new_pos]
                self.grid.cells[self.pos] = 0
                self.grid.cells[new_pos] = self.orients[self.orientation]
                self.pos = new_pos
                self.history[0].append(self.pos[0])
                self.history[1].append(self.pos[1])
                # Death:
                if tile_after_move == 3:
                    self.alive = False
                    return False
                return True
            else:
                return False

    def rotate(self, dir):
        current = list(self.orients.keys()).index(self.orientation)
        if dir == 'r':
            self.orientation = list(self.orients.keys())[(current + 1) % 4]
        elif dir == 'l':
            self.orientation = list(self.orients.keys())[current - 1]
        self.grid.cells[self.pos] = self.orients[self.orientation]


class Grid:
    def __init__(self, n_cols, n_rows):
        self.n_rows = n_rows
        self.n_cols = n_cols
        # Building the boundary of the grid:
        self.cells = np.ones((n_cols, n_rows))
        self.cells[0, :] = self.cells[-1, :] = -1
        self.cells[:, 0] = self.cells[:, -1] = -1

    def put_obstacle(self, x0, x1, y0, y1, from_edge=1):
        self.cells[max(x0, from_edge):min(x1 + 1, self.n_cols - from_edge),
        max(y0, from_edge):min(y1 + 1, self.n_rows - from_edge)] = -2

    def put_singular_obstacle(self, x, y):
        self.cells[x][y] = -2

    def put_singular_goal(self, x, y):
        self.cells[x][y] = 2

    def put_singular_death(self, x, y):
        self.cells[x][y] = 3


def generate_grid(n_cols, n_rows):
    # Placeholder function used to generate a grid.
    # Select an empty grid file in the user interface and add code her to automatically fill it.
    # Look at grid_generator.py for inspiration.
    grid = Grid(n_cols, n_rows)
    return grid
