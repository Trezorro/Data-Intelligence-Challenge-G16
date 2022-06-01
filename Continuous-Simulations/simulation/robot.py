"""Robot.

The robot that will vacuum the map.
"""
from typing import Tuple

from rectangle import Rectangle
import math
from random import random
from simulation.grid import Grid


class Robot(Rectangle):
    def __init__(self, x: float, y: float, angle: float = 0, vision=None,
                 battery_drain_p: float = 1):
        """Robot class which can vacuum.

        The robot is based off of the retangle class. Its shape is always
        orthogonal to the grid itself, but it can move at different angles and
        always has a fixed size.

        Args:
            x: x-coordinate of the center of the robot
            y: y-coordiante of the center of the robot
            angle: Angle that the robot is currently facing in degrees.
            vision: I'm not sure how we want vision to work yet.
            battery_drain_p: Probability of battery_level draining. Value must be
                between [0, 1]
        """
        assert 0 <= battery_drain_p <= 1, "Battery drain probability " \
                                          "outside of [0, 1]"
        # Make sure robot does not spawn outside of the grid
        x = max(x, 25)
        y = max(y, 25)

        robot_r = 25  # "Radius" of the robot, i.e. half the side length.

        super(Robot, self).__init__(x - robot_r, y - robot_r,
                                    x + robot_r, y + robot_r)

        self.angle = angle
        self.battery_level = 100
        self.alive = True
        self.vision = vision
        self.battery_drain_p = battery_drain_p
        self.history = []

    def move(self, distance: float, grid: Grid) -> Tuple[bool, bool]:
        """Moves the robot the given distance.

        Args:
            distance: The distance to move the robot.
            grid: The grid the robot is on

        Returns:
            Two boolean values:
                1. Whether the robot was able to succesfully move
                2. Whether the robot died at the end of the move.
        """
        # New potential position
        new_x1 = self.x1 + distance * math.cos(self.angle)
        new_y1 = self.y1 + distance * math.sin(self.angle)
        new_x2 = self.x2 + distance * math.cos(self.angle)
        new_y2 = self.y2 + distance * math.sin(self.angle)

        # Check if new position is out of bounds
        if (new_x1 > 0
                or new_y1 > 0
                or new_x2 < grid.width
                or new_y2 < grid.height):
            # Robot could not move in the requested direction
            self.x1 = new_x1
            self.x2 = new_x2
            self.y1 = new_y1
            self.y2 = new_y2
            robot_moved = True
        else:
            robot_moved = False

        # ~20x faster than np.random.binomial()
        do_battery_drain = random() > self.battery_drain_p

        if do_battery_drain:
            self.battery_level -= 1

        if self.battery_level <= 0:
            self.alive = False

        return robot_moved, not self.alive

    def rotate(self, angle: float):
        """ Rotates the robot by the given amount

        Args:
            angle: Amount to rotate the robot by, in degrees.
        """
        # Rotate the robot
        self.angle += angle
        self.angle %= 360
