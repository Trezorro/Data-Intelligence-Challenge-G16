from environment import Grid
import os
import numpy as np
import pickle

PATH = os.getcwd()

for k in range(5):
    # Get a grid with a random height and width:
    height = np.random.randint(10, 20)
    width = np.random.randint(10, 20)
    grid = Grid(width, height)
    # Create the corridor:
    corr_y0 = int(height / 2)
    corr_y1 = int(height / 2) + 2

    n_rooms = 5
    # Get upper rooms:
    rooms = np.random.randint(2, n_rooms)
    for i in range(0, width, width//rooms):
        grid.put_obstacle(x0=i, x1=i + (width//rooms) -2, y0=corr_y0, y1=corr_y0)
    for i in range(0, rooms-1):
        grid.put_obstacle(x0=(i+1) * int(width / rooms), x1=(i+1) *int(width / rooms), y0=1, y1=corr_y0)

    # Get lower rooms:
    rooms = np.random.randint(2, n_rooms)
    for i in range(0, width, width // rooms):
        grid.put_obstacle(x0=i, x1=i + (width // rooms) - 2, y0=corr_y1, y1=corr_y1)
    for i in range(0, rooms - 1):
        grid.put_obstacle(x0=(i + 1) * int(width / rooms), x1=(i + 1) * int(width / rooms), y0=corr_y1, y1=height)

    name = 'example-random-house'
    pickle.dump(grid, open(f'{PATH}/grid_configs/{name}-{k}.grid', 'wb'))