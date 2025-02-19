from lux import game
import random
from typing import List, Set, Tuple

from .constants import Constants

DIRECTIONS = Constants.DIRECTIONS


class Position:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __sub__(self, pos: 'Position') -> int:
        return abs(pos.x - self.x) + abs(pos.y - self.y)

    def distance_to(self, pos: 'Position'):
        """
        Returns Manhattan (L1/grid) distance to pos
        """
        return self - pos

    def is_adjacent(self, pos: 'Position'):
        return (self - pos) <= 1

    def __eq__(self, pos: 'Position') -> bool:
        return self.x == pos.x and self.y == pos.y

    def equals(self, pos: 'Position'):
        return self == pos

    def translate(self, direction, units) -> 'Position':
        if direction == DIRECTIONS.NORTH:
            return Position(self.x, self.y - units)
        elif direction == DIRECTIONS.EAST:
            return Position(self.x + units, self.y)
        elif direction == DIRECTIONS.SOUTH:
            return Position(self.x, self.y + units)
        elif direction == DIRECTIONS.WEST:
            return Position(self.x - units, self.y)
        elif direction == DIRECTIONS.CENTER:
            return Position(self.x, self.y)

    def direction_to(self, target_pos: 'Position') -> DIRECTIONS:
        """
        Return closest position to target_pos from this position
        """
        check_dirs = [
            DIRECTIONS.NORTH,
            DIRECTIONS.EAST,
            DIRECTIONS.SOUTH,
            DIRECTIONS.WEST,
        ]
        random.shuffle(check_dirs)  # Shuffle for randomization

        closest_dist = self.distance_to(target_pos)
        closest_dir = DIRECTIONS.CENTER
        for direction in check_dirs:
            newpos = self.translate(direction, 1)
            dist = target_pos.distance_to(newpos)
            if dist < closest_dist:
                closest_dir = direction
                closest_dist = dist
        return closest_dir

    def __str__(self) -> str:
        return f"({self.x}, {self.y})"

    def __iter__(self):
        for i in (self.x, self.y):
            yield i
