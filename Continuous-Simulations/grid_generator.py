from typing import Optional, Union
from tqdm import trange
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
    # Add borders of 1 to grid
    grid[0, :] = 1
    grid[grid_size-1, :] = 1
    grid[:, 0] = 1
    grid[:, grid_size-1] = 1
    grid[grid_size-1, corridor_start+1:corridor_end] = 3  # Put death tiles in the beginning of the corridor
    return grid


class GridBuilder:
    def __init__(
            self, 
            grid_size: int = 24, 
            room_sizes: Union[tuple, list] = ((8, 7), (7, 6), (7, 6), (6, 5), (4, 2), (3, 2), (2, 1), (2, 1)),
    ):
        """The GridBuilder can generate grids with specified rooms 
        allocated randomly within the allocated space. 
        Default setting: grid 24x24, 5 rooms - living room, kitchen, toilet, bedroom, garden exit 
        """
        self.grid = get_grid_with_corridor(grid_size)
        self.room_sizes = room_sizes
        
    def generate_grid(self, room_sizes: Optional[list] = None):
        """Generates a grid with specified rooms allocated randomly within the allocated space.
        :param: room_sizes: list of room sizes, if not specified, default sizes are used
        """
        if room_sizes is None:
            room_sizes = self.room_sizes

        for room in room_sizes:
            if room[0] > 4:
                self.place_room_randomly(room, death_room=False)
            else:
                self.place_room_randomly(room, death_room=True)

        self.grid = self.grid.astype(dtype=np.uint8)
        return self.grid

    def place_room_randomly(self, room_size: tuple, death_room: bool = False):
        """ Finds possible top left corners for a room with given size
        and randomly selects one, adding the borders to grid as walls
        """
        possible_grid = np.zeros((self.grid.shape[0], self.grid.shape[1]))
        for h in range(self.grid.shape[0]):
            for w in range(self.grid.shape[1]):
                if h + room_size[0] < self.grid.shape[0] and w + room_size[1] < self.grid.shape[1] and \
                        np.sum(self.grid[h+1:h + room_size[0], w+1:w + room_size[1]]) == 0:
                    possible_grid[h, w] += 2  # Cannot use negative because sum will not work

        # Find random placement for top-left corner of the room
        all_possible_idx = np.argwhere(possible_grid > 1)
        if len(all_possible_idx) > 0:
            random_idx = random.choice(all_possible_idx)
            top_left_corner_x = random_idx[0]
            top_left_corner_y = random_idx[1]

            # for idx in all_possible_idx:
            #     self.grid[(idx[0], idx[1])] -= 2
            if not death_room:
                self.grid[top_left_corner_x:top_left_corner_x + room_size[0] + 1, top_left_corner_y] = 1
                self.grid[top_left_corner_x, top_left_corner_y:top_left_corner_y + room_size[1] + 1] = 1
                self.grid[top_left_corner_x:top_left_corner_x + room_size[0] + 1, top_left_corner_y + room_size[1]] = 1
                self.grid[top_left_corner_x + room_size[0], top_left_corner_y:top_left_corner_y + room_size[1] + 1] = 1
            else:
                self.grid[
                    max(top_left_corner_x, 1):min(max(top_left_corner_x, 1) + room_size[0], len(self.grid) - 2),
                    max(top_left_corner_y, 1):min(max(top_left_corner_y, 1) + room_size[1], len(self.grid) - 2)
                ] = 3

            # Create doors
            if not death_room:
                self.grid[top_left_corner_x + room_size[0] // 2,  max(top_left_corner_y, 1)] = 0
                self.grid[top_left_corner_x + room_size[0] // 2 + 1, max(top_left_corner_y, 1)] = 0
                self.grid[max(top_left_corner_x, 1), top_left_corner_y + room_size[1] // 2] = 0
                self.grid[max(top_left_corner_x, 1), top_left_corner_y + room_size[1] // 2 + 1] = 0
                self.grid[
                    min(top_left_corner_x + room_size[0], len(self.grid) - 2),
                    top_left_corner_y + room_size[1] // 2
                ] = 0
                self.grid[
                    min(top_left_corner_x + room_size[0], len(self.grid) - 2),
                    top_left_corner_y + room_size[1] // 2 + 1
                ] = 0
                self.grid[
                    top_left_corner_x + room_size[0] // 2 + 1,
                    min(top_left_corner_y + room_size[1], len(self.grid) - 2)
                ] = 0
                self.grid[
                    top_left_corner_x + room_size[0] // 2,
                    min(top_left_corner_y + room_size[1], len(self.grid) - 2)
                ] = 0


if __name__ == '__main__':
    for i in trange(100000):
        try:
            g = GridBuilder().generate_grid()
        except Exception as e:
            print(e)
            break
