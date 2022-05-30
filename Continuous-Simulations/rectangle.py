"""Rectangle.

Rectangle for the robot to crash into/clean
"""


class Rectangle:
    def __init__(self, x1, y1, x2, y2, rectangle_type):
        """Initialize the rectangle.

        Args:
            x1: Upper left x-coordinate of the rectangle.
            y1: Upper left y-coordinate of the rectangle.
            x2: Lower right x-coordinate of the rectangle.
            y2: Lower right y-coordinate of the rectangle.
            rectangle_type: Type of the rectangle. One of
                ['obstacle', 'dirt', 'goal', 'death', 'robot'].
        """
        self.x1 = x1,
        self.y1 = y1,
        self.x2 = x2,
        self.y2 = y2
        self.x_size = x2 - x1
        self.y_size = y2 - y1

        if rectangle_type not in ("obstacle", "dirt", "goal", "death", "robot"):
            raise ValueError("Invalid rectangle type")

        self.rectangle_type = rectangle_type

    def intersect(self, other):
        """Check if the rectangle intersects with another rectangle.

        Args:
            other: Another rectangle to check intersection with.
        """
        intersecting = not (self.x2 <= other.x1
                            or self.x1 >= other.x2
                            or self.y2 <= other.y1
                            or self.y1 >= other.y2)
        inside = (other.x1 >= self.x1
                  and other.x2 <= self.x2
                  and other.y1 >= self.y1
                  and other.y2 <= self.y2)
        return intersecting or inside

    def update_pos(self, x, y):
        """ Update the position of the rectangle.

        Args:
            x: New upper right x-coordinate of the rectangle.
            y: New upper right y-coordinate of the rectangle.

        """
        self.x1, self.x2 = x, x + self.x_size
        self.y1, self.y2 = y, y + self.y_size

