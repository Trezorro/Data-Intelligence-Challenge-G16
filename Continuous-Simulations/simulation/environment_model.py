"""Environment Model.

The internal environment used to maintain an awareness of where everything is.
"""
import math
from random import random
from typing import List, Dict, Tuple

import numpy as np
from pygame import Rect


class EnvironmentModel:
    def __init__(
            self,
            grid: np.ndarray,
            num_obstacles: int,
            num_dirt: int,
            cell_size: int = 64,
            battery_drain: float = 0.25,
            agent_width: int = 48
    ):
        """A very simple environment model which holds data about the world.

        The world is represented as a continuous grid with the size based on
        the size of the provided grid.

        Args:
            grid: A (usually) 24x24 array containing where walls are. Rows is x,
                cols is y.
            num_obstacles: The number of obstacles to place.
            num_dirt: The number of dirt particles to place.
            cell_size: The size of each grid cell in number of pixels
            battery_drain: The amount of battery drained with every step for
                the agent.
            agent_width: The width of the agent on the world.
        """
        # Set environment parameters
        self.cell_size = cell_size
        self.battery_drain = battery_drain
        self.grid = grid
        self.grid_size = grid.shape[0]
        self.world_size = self.grid_size * self.cell_size

        self.floor_rects: List[Rect] = self._grid_to_rects(grid, 0)
        self.wall_rects: List[Rect] = self._grid_to_rects(self.grid, 1)
        self.death_rects: List[Rect] = self._grid_to_rects(self.grid, 3)

        self.obstacle_rects: List[Rect] = []
        self.dirt_rects: List[Rect] = []

        self.agent_rect = Rect(0, 0, 0, 0)  # Initialize with a dummy value
        self.agent_angle = 0  # Angle of agent in degrees, 0 being north
        self.agent_battery = 100.0
        self.agent_is_alive: bool = True
        self.agent_width = agent_width

        self.previous_observation = None

        self._place_obstacles(num_obstacles)
        self._place_dirt(num_dirt)
        self._place_agent()

    def _cell_to_rect(self, cell: Tuple[int, int]) -> Rect:
        """Converts a cell to a rectangle.

        Args:
            cell: The cell to convert as (x, y) tuple.

        Returns:
            A rectangle representing the cell.
        """
        return Rect(int(cell[0] * self.cell_size),
                    int(cell[1] * self.cell_size),
                    self.cell_size,
                    self.cell_size)

    def _grid_to_rects(self, grid: np.ndarray, val) -> List[Rect]:
        """Places walls and death tiles as a list of Rect objects.

        Walls are represented as a bunch of rectangular objects in the world,
        NOT the grid. Each wall square has a size of scalar x scalar.

        Args:
            grid: The grid to use to generate the rectangles.
            val: The value to match to figure out the rectangles to generate.
                For example, to get walls, val should be 1 and for death tiles,
                val should be 2.

        Returns:
             A list of rectangles representing some wall or death tile objects.
        """
        idxs = np.nonzero(grid == val)
        positive_cells = list(zip(idxs[0], idxs[1]))

        return [self._cell_to_rect(cell) for cell in positive_cells]

    def _place_thing(self, width: float, height: float) -> Rect:
        """Places a thing somewhere on the grid, respecting walls.

        Args:
            width: The width of the object.
            height: The height of the object.

        Returns:
            The random position somewhere on the grid, respecting all other
            already placed objects.
        """
        thing_placed = False
        while not thing_placed:
            # Keep trying until a position that is not a wall is found.
            position = np.random.random(2) * self.grid_size * self.cell_size
            rect = Rect(position[0], position[1], width, height)

            # Check for collisions
            collisions_dict = self._check_colisions(rect)
            # Check that all produced collision lists are of length 0, i.e.,
            # that all types of collisions are avoided
            thing_placed = all([len(v) == 0 for v in collisions_dict.values()])

        return rect

    def _place_obstacles(self, num_obstacles: int):
        """Places the given number of obstacles on the grid.

        Obstacles have a minimum side length of 0.5x wall and maximum side
        length of 3.5x wall

        Args:
            num_obstacles: The number of obstacles to be placed.
        """
        for _ in range(num_obstacles):
            width = (random() * 3. * self.cell_size) + 0.5
            height = (random() * 3. * self.cell_size) + 0.5

            self.obstacle_rects.append(self._place_thing(width, height))

    def _place_dirt(self, num_dirt: int):
        """Places the given number of dirt particles on the grid.

        Dirt is represented as simply points on the grid.

        Args:
            num_dirt: The number of dirt particles to be placed.
        """
        for _ in range(num_dirt):
            # We do this sequentially so that each previous dirt object will
            # have an effect on the position of the next dirt object and not
            # overlap.
            self.dirt_rects.append(self._place_thing(1, 1))

    def _place_agent(self):
        """Places the agent somewhere on the world, respecting existing objects.

        The agent should always be placed last so it doesn't start off inside
        an obstacle or on top of dirt.
        """
        self.agent_rect = self._place_thing(width=self.cell_size / 2,
                                            height=self.cell_size / 2)

    def _check_colisions(self, rect: Rect) -> Dict[str, list]:
        """Tests if the provided rectangle collides with anything in the world.

        Args:
            rect: The object to test

        Returns:
             A dictionary with the list of indices of collided objects.
                An empty list means no collisions. The keys are
                ['walls', 'obstacles', 'dirt', 'death'].
                An example returned dictionary would be
                {"walls": [], "obstacles": [2], "dirt": [1, 5], "death": []}
        """
        return {"walls": rect.collidelistall(self.wall_rects),
                "obstacles": rect.collidelistall(self.obstacle_rects),
                "dirt": rect.collidelistall(self.dirt_rects),
                "death": rect.collidelistall(self.death_rects)}

    def _drain_battery(self) -> bool:
        """Drains battery by a certain amount.

        Returns:
            If the agent is still alive or not after the battery drain.
        """
        self.agent_battery -= self.battery_drain
        return self.agent_battery > 0

    def rotate_agent(self, angle: int):
        """Rotates the agent but the desired amount.

        Args:
            angle: How much to rotate the agent by. NOT the new angle of the
                agent
        """
        if self.agent_is_alive:
            self.agent_angle = (self.agent_angle + angle) % 360

    def move_agent(self, distance: int) -> Tuple[dict, np.ndarray]:
        """Moves the agent by the given distance and checks for colisions.

        Sets the observation dict which is a dictionary containing keys
        [move_succeeded, hit_wall, hit_obstacle, hit_dirt, hit_death, is_alive].
        `hit_dirt` is an int representing the number of dirt objects hit during
        the move. Every other key has a value of a bool.

        The world observation is returned as a grid array with 3-dimensions
        where each z axis contains 0 or 1 values. Each z axis represents a
        different types of object in the world. A value of 1 means that
        something of that object type exists in that grid square.
            0: Walls
            1: Death
            2: Obstacles
            3: Dirt
            4: Visited
            5: Fog of war

        Args:
            distance: The distance to move the agent by.

        Returns:
            The event dictionary and the robot's new observation of the world.
        """
        move_succeeded = True
        hit_wall = False
        hit_obstacle = False
        hit_dirt = 0
        hit_death = False

        if self.agent_is_alive:
            # Moves the agent by the given distance respecting its current angle
            heading = np.array([math.cos(self.agent_angle / 180 * math.pi),
                                math.sin(self.agent_angle / 180 * math.pi)])
            move_by = (heading * distance).astype(int)
            move_by = (int(move_by[0]), int(move_by[1]))
            self.agent_rect = self.agent_rect.move(move_by[0], move_by[1])

            # Check if agent is still alive after draining the battery. Every
            # move should drain the battery since even if a robot bumps into a
            # wall, it's still expending energy to do so.
            self.agent_is_alive = self._drain_battery()

            # Check for colisions
            collisions = self._check_colisions(self.agent_rect)

            # Check if we hit a wall
            if len(collisions["walls"]) > 0:
                move_succeeded = False
                hit_wall = True

            # check if we hit an obstacle
            elif len(collisions["obstacles"]) > 0:
                move_succeeded = False
                hit_obstacle = True

            # Check if we hit a death tile
            elif len(collisions["death"]) > 0:
                move_succeeded = False
                hit_death = True
                self.agent_is_alive = False

            if move_succeeded:
                hit_dirt = len(collisions["dirt"])
                if hit_dirt > 0:
                    # Remove vacuumed up dirt
                    new_dirt_rects = [dirt
                                      for i, dirt in enumerate(self.dirt_rects)
                                      if i not in collisions["dirt"]]
                    self.dirt_rects = new_dirt_rects
            else:
                self.agent_rect = self.agent_rect.move(-move_by[0], -move_by[1])
        else:
            move_succeeded = False

        events = {
            "move_succeeded": move_succeeded,
            "hit_wall": hit_wall,
            "hit_obstacle": hit_obstacle,
            "hit_dirt": hit_dirt,
            "hit_death": hit_death,
            "is_alive": self.agent_is_alive
        }
        # TODO: maintain fog of war and obstacle channels
        # add visible cells to fog of war 
        # add visible obstacles to obstacle channel
        world_observation = None

        self.previous_observation = events, world_observation

        return events, world_observation

    def _cell_visible(self, cell_x: int, cell_y: int) -> bool:
        """Checks if the cell is visible to the agent.

        Args:
            cell: The cell to check.

        Returns:
            True if the cell is visible, False otherwise.
        """
        agent_angle_rad = self.agent_angle * math.pi / 180
        agent_direction_vec = np.array([math.cos(agent_angle_rad),
                                        math.sin(agent_angle_rad)])

        # Figure out cell side positions
        cell_rect = self._cell_to_rect((cell_x, cell_y))
        sides = [
            np.array(cell_rect.midtop),
            np.array(cell_rect.midleft),
            np.array(cell_rect.midbottom),
            np.array(cell_rect.midright),
        ]
        for side_center in sides:
            line_of_sight_vec = side_center - np.array(self.agent_rect.center)
            if line_of_sight_vec == np.zeros(2):  # Agent is exactly on the side
                return True
            norm = np.sqrt(line_of_sight_vec @ line_of_sight_vec)
            if agent_direction_vec @ (
                    line_of_sight_vec / norm) < 0.7:  # Outside view cone of +-90 degrees
                continue
            for wall_rect in self.wall_rects:
                if wall_rect.clipline(self.agent_rect.center,
                                      side_center) != tuple():
                    break  # line of sight blocked by a wall
            return True
        return False
