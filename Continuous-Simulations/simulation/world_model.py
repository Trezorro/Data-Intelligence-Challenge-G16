"""Environment Model.

The internal environment used to maintain an awareness of where everything is.
"""
import math
from random import random
from typing import List, Dict, Tuple

import numpy as np
from pygame import Rect


class WorldModel:
    def __init__(
            self,
            grid: np.ndarray = np.zeros((24, 24)),
            num_obstacles: int = 3,
            num_dirt: int = 100,
            cell_size: int = 64,
            battery_drain: float = 0.25,
            agent_width: int = 60,
            agent_speed=60,
            slam_accuracy: float = 0.7
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
            slam_accuracy: Simulated accuracy of the SLAM algorithm used to
                figure out where all the walls are. How this works is when
                walls from the grid are added to the world, it has a percentage
                chance of being added to wall_rects and a 1-p chance of being
                added as an obstacle instead. Must be between 0 and 1.

        """
        # Set environment parameters
        self.cell_size = cell_size
        self.battery_drain = battery_drain
        self.grid = grid
        self.grid_size = grid.shape[0]
        self.world_size = self.grid_size * self.cell_size
        self.start_dirtiness = num_dirt

        self.occluding_walls: List[Rect] = []  # actual walls
        self.wall_rects: List[Rect] = []  # walls that the robot detected using SLAM
        self.death_rects: List[Rect] = []

        self.obstacle_rects: List[Rect] = []
        self.obstacle_grid = np.zeros_like(self.grid, dtype=float)
        self.dirt_rects: List[Rect] = []
        self.dirt_grid = np.zeros_like(self.grid, dtype=float)

        self.agent_rect = Rect(0, 0, 0, 0)  # Initialize with a dummy value
        self.agent_angle = 0  # Angle of agent in degrees, 0 being north
        self.agent_battery = 100.0
        self.agent_is_alive: bool = True
        self.agent_width = agent_width
        self.agent_speed = agent_speed

        self.visited_grid = np.zeros_like(self.grid, dtype=float)
        self.fow_grid = np.zeros_like(self.grid, dtype=float)

        self._cell_rects = [[self._cell_to_rect((x, y))
                             for y in range(self.grid.shape[0])]
                            for x in range(self.grid.shape[1])]

        self._place_walls_and_death(slam_accuracy)
        self._place_obstacles(num_obstacles)
        self._place_agent()
        self._place_dirt(num_dirt)

        self.last_observation = self._get_world_observation()

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
            collisions_dict = self._check_collisions(rect)
            # Check that all produced collision lists are of length 0, i.e.,
            # that all types of collisions are avoided
            thing_placed = all([len(v) == 0 for v in collisions_dict.values()])

        return rect

    @staticmethod
    def _calculate_overlap(rect: Rect, other: Rect) -> float:
        """Calculates overlap between two rectangles.

        Both rects must have an area > 0.

        Returns:
            The proportion of rect which is filled with other.
        """
        int_x0 = max(rect.x, other.x)
        int_x1 = min(rect.x + rect.width, other.x + other.width)
        int_y0 = max(rect.y, other.y)
        int_y1 = min(rect.y + rect.height, other.y + other.height)
        int_width = max(0, int_x1 - int_x0)
        int_height = max(0, int_y1 - int_y0)
        int_area = int_width * int_height

        return min(1., float(int_area) / float(rect.width * rect.height))

    def _place_walls_and_death(self, slam_accuracy: float):
        """Places walls and death tiles with a given accuracy.

        Args:
            slam_accuracy: A value between 0 and 1.
        """
        # For walls (1) and death (3)
        for val in (1, 3):
            idxs = np.nonzero(self.grid == val)
            positive_cells = list(zip(idxs[0], idxs[1]))
            for cell in positive_cells:
                rect = self._cell_to_rect(cell)
                if val == 1:
                    self.occluding_walls.append(rect)
                if random() < slam_accuracy:
                    if val == 1:
                        self.wall_rects.append(rect)
                    else:
                        self.death_rects.append(rect)
                else:
                    self.obstacle_rects.append(rect)

    def _place_obstacles(self, num_obstacles: int):
        """Places the given number of obstacles on the grid.

        Obstacles have a minimum side length of 0.5x wall and maximum side
        length of 3.5x wall

        Args:
            num_obstacles: The number of obstacles to be placed.
        """
        # TODO: increase minimum obstacle size
        for _ in range(num_obstacles):
            width = (random() * 3. * self.cell_size) + 0.5
            height = (random() * 3. * self.cell_size) + 0.5

            obstacle = self._place_thing(width, height)
            for x, row in enumerate(self._cell_rects):
                for y, cell in enumerate(row):
                    overlap = self._calculate_overlap(cell, obstacle)
                    self.obstacle_grid[x, y] += overlap

            self.obstacle_rects.append(obstacle)

        # Normalize obstacle_grid when done
        self.obstacle_grid -= np.min(self.obstacle_grid)
        self.obstacle_grid /= np.max(self.obstacle_grid)

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
            dirt = self._place_thing(1, 1)
            for x, row in enumerate(self._cell_rects):
                for y, cell in enumerate(row):
                    if cell.contains(dirt):
                        self.dirt_grid[x, y] += 1
            self.dirt_rects.append(dirt)

        # Normalize dirt_grid when done
        self.dirt_grid -= np.min(self.dirt_grid)
        self.dirt_grid /= np.max(self.dirt_grid)

    def _place_agent(self):
        """Places the agent somewhere on the world, respecting existing objects.

        The agent should always be placed last, so it doesn't start off inside
        an obstacle or on top of dirt.
        """
        self.agent_rect = self._place_thing(width=self.agent_width,
                                            height=self.agent_width)

    def _check_collisions(self, rect: Rect, check_walls=True) -> Dict[str, list]:
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
        collisions = {"obstacles": rect.collidelistall(self.obstacle_rects),
                      "dirt": rect.collidelistall(self.dirt_rects),
                      "death": rect.collidelistall(self.death_rects)}
        if check_walls:
            collisions["walls"] = rect.collidelistall(self.occluding_walls)

        return collisions

    def _drain_battery(self, move_amount=1.0) -> float:
        """Drains battery by a certain amount.

        Returns:
            If the agent is still alive or not after the battery drain.
        """
        drain_amount = self.battery_drain * move_amount
        self.agent_battery -= drain_amount
        return drain_amount

    def rotate_agent(self, angle: int):
        """Rotates the agent but the desired amount.

        Args:
            angle: How much to rotate the agent by. NOT the new angle of the
                agent
        """
        if self.agent_is_alive:
            self.agent_angle = (self.agent_angle + angle) % 360

    def move_agent(self, action_distance: int, rotate_angle: int) -> Tuple[dict, np.ndarray]:
        """Moves the agent by the given distance and checks for colisions.

        Sets the observation dict which is a dictionary containing keys
        [move_succeeded, hit_wall, hit_obstacle, hit_dirt, hit_death, is_alive].
        `hit_dirt` is an int representing the number of dirt objects hit during
        the move. Every other key has a value of a bool.

        The world observation is returned as a grid array with 3-dimensions
        where each z axis contains 0 or 1 values. Each z axis represents a
        different types of object in the world. A value of 1 means that
        something of that object type exists in that grid square.
            0: Walls / death
            1: Obstacles
            2: Dirt
            3: Visited
            4: Fog of war

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
            self.rotate_agent(rotate_angle)

            # Moves the agent by the given distance respecting its current angle
            heading = np.array([math.cos(self.agent_angle / 180 * math.pi),
                                math.sin(self.agent_angle / 180 * math.pi)])

            for completion_ratio in [r / 100 for r in range(100, -1, -5)]:
                moved_distance = action_distance * completion_ratio
                move_by = (heading * moved_distance).astype(int)
                move_by = (int(move_by[0]), int(move_by[1]))
                next_position_rect = self.agent_rect.move(move_by[0], move_by[1])

                # Check for collisions

                # Check if we hit a wall or obstacle
                if len(next_position_rect.collidelistall(self.occluding_walls)) > 0:
                    hit_wall = True
                if len(next_position_rect.collidelistall(self.obstacle_rects)) > 0:
                    hit_obstacle = True
                if hit_wall or hit_obstacle:
                    move_succeeded = False
                    continue  # backtrack a bit

                # no wall or obstacle, move successful! Check for dirt and death.
                self.agent_rect = next_position_rect
                break


            drain_amount = self._drain_battery(move_amount=action_distance / self.agent_speed + abs(rotate_angle%360) /360) 
            self.agent_is_alive = self.agent_battery > 0

            collisions = self._check_collisions(self.agent_rect, check_walls=False)
            # Check if we hit a death tile
            if len(collisions["death"]) > 0:
                move_succeeded = False
                hit_death = True
                self.agent_is_alive = False
            # Check if agent is still alive after draining the battery. Every
            # move should drain the battery since even if a robot bumps into a
            # wall, it's still expending energy to do so.

            hit_dirt = len(collisions["dirt"])
            if hit_dirt > 0:
                # Remove vacuumed up dirt
                new_dirt_rects = [dirt
                                  for i, dirt in enumerate(self.dirt_rects)
                                  if i not in collisions["dirt"]]
                self.dirt_rects = new_dirt_rects

            # Set the current grid cell as visited
            x_pos = int(self.agent_rect.x / self.cell_size)
            y_pos = int(self.agent_rect.y / self.cell_size)
            self.visited_grid[x_pos, y_pos] = 1
        else:
            move_succeeded = False
            drain_amount = 0
            

        events = {
            "move_succeeded": int(move_succeeded),
            "hit_wall": int(hit_wall),
            "hit_obstacle": int(hit_obstacle),
            "hit_dirt": int(hit_dirt),
            "hit_death": int(hit_death),
            "is_alive": int(self.agent_is_alive),
            "drain_amount": drain_amount
        }

        self.last_observation = self._get_world_observation()

        return events, self.last_observation

    def _get_world_observation(self) -> np.ndarray:
        """Gets the world observation.

        A value of 1 means that cell is completely filled with the object of
        that type. Uniquely, for walls/death, a 1 represents a wall and a 3
         represents a death tile.
            0: Walls / death
            1: Obstacles
            2: Dirt
            3: Visited
            4: Fog of war
        """
        # Calculate fog of war
        for x, row in enumerate(self._cell_rects):
            for y, cell in enumerate(row):
                if self.fow_grid[x, y] == 0.:
                    # Only check if a cell has not been visited yet
                    self.fow_grid[x, y] = 1. * self._cell_visible(x, y)

        dirt_with_fow = self.dirt_grid * self.fow_grid
        obstacles_with_fow = self.obstacle_grid * self.fow_grid

        new_shape = [1, self.grid.shape[0], self.grid.shape[1]]
        obs = np.concatenate([
            self.grid.reshape(new_shape),  # 0: walls/death
            obstacles_with_fow.reshape(new_shape),  # 1: obstacles
            dirt_with_fow.reshape(new_shape),  # 2: dirt
            self.visited_grid.reshape(new_shape),  # 3: visited
            self.fow_grid.reshape(new_shape)  # 4: fog of war
        ], axis=0).astype(float)

        return obs

    def _cell_visible(self, cell_x: int, cell_y: int) -> bool:
        """Checks if the cell is visible to the agent.

        Args:
            cell_x: the x coordinate of the cell to check.
            cell_y: the y coordinate of the cell to check.

        Returns:
            True if the cell is visible, False otherwise.
        """
        agent_center = np.array(self.agent_rect.center)
        agent_angle_rad = self.agent_angle * math.pi / 180
        agent_direction_vec = np.array([math.cos(agent_angle_rad),
                                        math.sin(agent_angle_rad)])

        # Figure out cell side positions
        cell_rect = self._cell_rects[cell_x][cell_y]
        sides = [
            cell_rect.midtop,
            cell_rect.midleft,
            cell_rect.midbottom,
            cell_rect.midright,
        ]
        for side_center in sides:
            line_of_sight_vec = np.array(side_center) - agent_center
            if not line_of_sight_vec.any():  # Agent is exactly on the side
                return True
            los_norm = np.sqrt(line_of_sight_vec @ line_of_sight_vec)
            if agent_direction_vec @ (line_of_sight_vec / los_norm) < 0.7:
                # Outside view cone of +-90 degrees
                continue
            no_ocludes = True
            for wall_rect in self.occluding_walls:
                if wall_rect.clipline(self.agent_rect.center, side_center):
                    # line of sight blocked by a wall
                    no_ocludes = False
                    break
            if no_ocludes:
                # Found a side that is visible
                return True
        return False
