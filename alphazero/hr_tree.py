from typing import Dict, Optional
import numpy as np
from alphazero.interfaces import Board, Move


class TreeNode:

    def __init__(self, parent, key: str, action: Optional[Move], hrs, value, nsa, ucb, level=1, info=None):
        self.info = info
        self.parent = parent
        self.level = level
        self.key = key
        self.action = action
        self.hrs = hrs
        self.value = value if value is not None else 0
        self.ucb = ucb if ucb is not None else -float('inf')
        self.nsa = nsa
        self.children = {}

    def add_child(self, move: Move, board: Board, value, nsa, ucb, info: Dict):
        key = board.get_string_representation()
        child = TreeNode(self, key, move, board.get_stones(), value, nsa, ucb, self.level + 1, info)
        self.children[str(move)] = child
        return child


    def from_here(self, child):
        """
        :param child: a decendent TreeNode
        :returns: the sequence that leads from parent to child, if child is a true decendent. Throws ValueError if not.
        """
        the_leaf = child.hrs[len(self.hrs):]
        if self.hrs + the_leaf == child.hrs:
            return the_leaf
        else:
            raise ValueError(f"Can't construct {child} from {self}")

    def __str__(self):
        return f"{self.action}: v={np.round(self.value, 7)}, " \
               f"nsa = {self.nsa}, ucb={np.round(self.ucb, 7)}"

    __repr__ = __str__

    def plot(self, depth=100, level=0):
        if self.action is not None:
            str_rep = ("  " * level) + f"{self.action}: v={np.round(self.value, 7)}, " \
                      f"nsa = {self.nsa}, ucb={np.round(self.ucb, 7)}"
            print(str_rep)
        level += 1
        if level < depth:
            for c in self.children.values():
                c.plot(level=level)


