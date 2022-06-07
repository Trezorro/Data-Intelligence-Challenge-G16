"""Environment Model.

The internal environment used to maintain an awareness of where everything is.
"""
from typing import List, Dict, Tuple
from pygame import Rect

import numpy as np


class EnvironmentModel:
    def __init__(self, map: np.ndarray, num_obstacles: int, num_dirt: int):
        """A very simple environment model which holds data about the world.

        The world is represented as a grid 32x bigger than the map array. For
        example, if the provided map array is 24x24, then the world has an edge
        length of 24x32=768.

        Args:
            map: A (usually) 24x24 array containing where walls are.
        """
        self.scalar = 32
        self.battery_drain = 0.25
        self.map = map

        self.agent_loc = self._translate_to_world(self._place_thing())
        self.agent_angle = 0  # Angle of agent in degrees, 0 being north
        self.agent_battery = 100.0
        self.agent_is_alive = True
        self.agent_width = 24
        self.agent_bbox = self._get_agent_bbox()

        self.obstacle_locs = self._place_things(num_obstacles)
        self.dirt_locs = self._place_things(num_dirt)

    def _place_thing(self) -> np.ndarray:
        """Places a thing somewhere on the map, respecting walls.

        Returns:
            The random position somewhere on the grid, not world, respecting
            walls
        """
        thing_placed = False
        while not thing_placed:
            # Keep trying until a position that is not a wall is found.
            position = np.random.randint(0, self.map.shape[0], 2)
            thing_placed = self.map[tuple(position)] == 0
        return position.astype(int)

    def _translate_to_world(self, pos: np.ndarray,
                            randomize: bool = False) -> np.ndarray:
        """Scales the smaller map position to the world position.

        Args:
            randomize: Whether or not to randomize the position within a grid
                position. If False, then puts it in the center of the grid
                square.
        Returns:
            The scaled new position
        """
        pos *= self.scalar
        if randomize:
            translation = np.random.randint(0, 32, 2)
            pos += translation
        else:
            pos += 16
        return pos

    def _translate_to_grid(self, pos: np.ndarray) -> np.ndarray:
        """Scales a world location to a grid position.

        Args:
            pos: A position in the world
        """
        return (pos / self.scalar).astype(int)

    def _place_things(self, num_things) -> Dict[tuple, np.ndarray]:
        """Places things on the map as a dict"""
        grid_positions = [self._place_thing() for _ in range(num_things)]
        things = {tuple(pos): self._translate_to_world(pos, True)
                  for pos in grid_positions}
        return things

    def _get_agent_bbox(self):
        """Gets the current bbox of the agent."""

    def _drain_battery(self) -> bool:
        """Drains battery by a certain amount.

        Returns:
            If the agent is still alive or not after the battery drain.
        """
        self.agent_battery -= self.battery_drain
        return self.agent_battery <= 0

    def rotate_agent(self, angle: int):
        """Rotates the agent but the desired amount.

        Args:
            angle: How much to rotate the agent by. NOT the new angle of the
                agent
        """
        self.agent_angle = (self.agent_angle + angle) % 360

    def move_agent(self, distance) -> Tuple[bool, bool, bool, bool, bool]:
        """Moves the agent by the given distance and checks for colisions.

        Args:
            distance: The distance to move the agent by.

        Returns:
            A tuple containing bools representing
                [move_succeeded, hit_wall, hit_obstacle, hit_dirt, is_alive]
        """
        move_succeeded = True
        hit_wall = False
        hit_obstacle = False
        hit_dirt = False

        # Moves the agentby the given distance respecting its current angle
        heading = np.array([np.cos(self.agent_angle), np.sin(self.agent_angle)])
        movement = (heading * heading).astype(int)
        self.agent_loc += movement
        self.agent_bbox.move(movement[0], movement[1])

        # Check if agent is still alive after draining the battery. Every move
        # should drain the battery since even if a robot bumps into a wall, it's
        # still expending energy to do so.
        is_alive = self._drain_battery()

        agent_grid_pos = self._translate_to_grid(self.agent_loc)

        # Check for colisions
        if self.map[tuple(agent_grid_pos)] == 1:
            self.agent_loc -= (distance * heading).astype(int)
            hit_wall = True
            move_succeeded = False

        if tuple(agent_grid_pos) in self.obstacle_locs.keys():
            # Then there is an obstacle in the same grid square.
            # Do collision calculation
            raise NotImplementedError
        if tuple(agent_grid_pos) in self.dirt_locs.keys():
            # Do collision calculation for dirt
            raise NotImplementedError

        return move_succeeded, hit_wall, hit_obstacle, hit_dirt, is_alive

    def get_world_observation(self) -> np.ndarray:
        """Gets the world observation as an 3-dimensional grid array.

        The world observation is returned as a grid array with 3-dimensions
        where each z axis contains 0 or 1 values. Each z axis represents a
        different types of object in the world. A value of 1 means that something
        of that object type exists in that grid square.
            0: Walls
            1: Obstacles
            2: Dirt
            3: Visited
            4: Fog of war

        Returns:
            A 3-dimensional grid array
        """
        raise NotImplementedError
