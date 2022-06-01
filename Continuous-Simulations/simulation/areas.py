"""Areas.

Represents areas the robot should either drive towards or keep away from such as
dirty areas, goal areas, and death areas.
"""
from rectangle import Rectangle
from enum import Enum


class AreaType(Enum):
    DIRTY = 0
    GOAL = 1
    DEATH = 2


class Area(Rectangle):
    def __init__(self,
                 x1: float, y1: float,
                 x2: float, y2: float,
                 area_type: AreaType):
        """Areas which are traversable for the robot.

        Args:
            x1: Upper left x-coordinate of the rectangle.
            y1: Upper left y-coordinate of the rectangle.
            x2: Lower right x-coordinate of the rectangle.
            y2: Lower right y-coordinate of the rectangle.
            area_type: Type of area this is supposed to represent.
        """
        super().__init__(x1, y1, x2, y2)
        self.area_type = area_type
