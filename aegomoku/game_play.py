from __future__ import annotations
from typing import List, Optional


class MoveNode:
    def __init__(self, move: int, parent: Optional[MoveNode]):
        self.move = move
        self.parent = parent
        self.children = []

    def __str__(self):
        return str(self.move)

    __repr__ = __str__


class GamePlay:
    """
    A utility class that tracks variations of game play to navigate interactively
    """

    def __init__(self, stones: List[int]):
        self.root = self.create_initial_play(stones)
        current = self.root
        while current is not None and len(current.children) > 0:
            current = current.children[0]
        self.current = current

    def create_initial_play(self, stones):
        if len(stones) == 0:
            return None

        root = MoveNode(stones[0], None)
        current = root
        for stone in stones[1:]:
            next_move = self.add_node(current, stone)
            current = next_move

        return root

    def add_node(self, parent: MoveNode, child: int):
        next_move = MoveNode(child, parent)
        next_move.parent = parent
        if parent:
            parent.children.append(next_move)
        else:
            self.root = next_move
        return next_move

    def bwd(self):
        if self.current is not None:
            self.current = self.current.parent
        return self.current

    def fwd(self, stone: int = None):

        if stone is not None:
            child = self.get_node(self.current, stone)
            if child is not None:
                self.current = child
            else:
                self.current = self.add_node(self.current, stone)

        else:
            # silent resilience
            # either choice or new or redundant
            if self.current is None:
                return None
            else:
                if len(self.current.children) == 0:
                    return self.current

            if len(self.current.children) == 1:
                self.current = self.current.children[0]
            else:
                raise ValueError(f"Multiple possible moves forward. Choose from {self.current.children}")

        return self.current

    def next_nodes(self):
        return self.current.children

    def cut(self):
        """cut the current branch and go back one move"""
        if self.current.parent is None:
            self.current = None
        else:
            current = self.current
            self.current = self.current.parent
            self.current.children.remove(current)

    @staticmethod
    def get_node(from_: MoveNode, stone: int):
        if from_ is None:
            return None
        for child in from_.children:
            if child.move == stone:
                return child
        return None

    def get_stones(self):
        stones = []
        current = self.current
        if current is None:
            return []
        while current is not None:
            stones.append(current.move)
            current = current.parent
        return list(reversed(stones))
