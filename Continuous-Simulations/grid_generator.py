import numpy as np
import random


def get_grid_with_corridor(grid_size: int):
    """Default grid is built as a 2D numpy array of size
    (grid_size, grid_size) and contains a random length corridor
    of size approximately (0.5grid_size, 3) positions
    somewhere in the center of the
    """
    grid = np.zeros((grid_size, grid_size))
    corridor_start = random.randint(4, grid_size-4)
    corridor_end = corridor_start + 3
    corridor_len = random.randint(4, grid_size-4)
    grid[corridor_len:grid_size, corridor_start] = 1
    grid[corridor_len:grid_size, corridor_end] = 1
    return grid


class GridBuilder:
    def __init__(
            self, 
            grid_size: int = 24, 
            n_rooms: int = 5, 
            room_sizes: tuple = ((8, 8), (8, 4), (4, 2), (4, 8), (3, 2))
    ):
        """The GridBuilder can generate grids with specified rooms 
        allocated randomly within the allocated space. 
        Default setting: grid 24x24, 5 rooms - living room, kitchen, toilet, bedroom, garden exit 
        """
        self.size = grid_size
        self.n_rooms = n_rooms
        self.room_sizes = room_sizes
        self.grid = get_grid_with_corridor(grid_size)
        
    def generate_grid(self):
        for room in range(self.n_rooms):
            self.place_room_randomly(self.room_sizes[room])
        print(self.grid)

    def place_room_randomly(self, room_size: tuple):
        """ Finds possible top left corners and randomly selects one,
        adding the borders to grid as walls
        """
        for h in range(self.grid.shape[0]):
            for w in range(self.grid.shape[1]):
                if h + room_size[0] < self.grid.shape[0] and w + room_size[1] < self.grid.shape[1] and \
                        np.sum(self.grid[h+1:h + room_size[0], w+1:w + room_size[1]]) == 0:
                    self.grid[h, w] += 2  # Cannot use negative because sum will not work

        # Find random placement for top-left corner of the room
        all_possible_idx = np.argwhere(self.grid > 1)
        if len(all_possible_idx) > 0:
            random_idx = random.choice(all_possible_idx)
            room_x = random_idx[0]
            room_y = random_idx[1]

            for idx in all_possible_idx:
                self.grid[(idx[0], idx[1])] = 0 if self.grid[(idx[0], idx[1])] == 2 else 1
                # Some walls are allowed to overlap, but we don't want to clear out the wall of another room accidentaly
            self.grid[room_x:room_x + room_size[0] + 1, room_y] = 1
            self.grid[room_x, room_y:room_y + room_size[1] + 1] = 1
            self.grid[room_x:room_x + room_size[0] + 1, room_y + room_size[1]] = 1
            self.grid[room_x + room_size[0], room_y:room_y + room_size[1] + 1] = 1


GridBuilder(16).generate_grid()

            
