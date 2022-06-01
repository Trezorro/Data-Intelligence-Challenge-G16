"""Obstacle.

Represents an obstacle in the continuous grid world.
"""
from rectangle import Rectangle
from enum import Enum


class ObstacleType(Enum):
    """Types of obstacles the robot might encounter."""
    WALL = 0


class Obstacle(Rectangle):
    def __init__(self,
                 x1: float, y1: float,
                 x2: float, y2: float,
                 obstacle_type: ObstacleType = ObstacleType.WALL):
        """Obstacles which are non traversable by the robot.

        Args:
            x1: Upper left x-coordinate of the rectangle.
            y1: Upper left y-coordinate of the rectangle.
            x2: Lower right x-coordinate of the rectangle.
            y2: Lower right y-coordinate of the rectangle.
            obstacle_type: The type of obstacle this is. Modular for future
                work.
        """
        super(Obstacle, self).__init__(x1, y1, x2, y2)

        self.obstacle_type = obstacle_type
