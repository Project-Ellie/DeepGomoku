from typing import List, Union

import numpy as np
from domoku.tools import GomokuTools as Gt


EMPTY_BOARDS = {
    n: np.rollaxis(
        np.array(2 * [[[0] * (n + 2)] * (n + 2)] + [[[1]*(n+2)] + n * [[1] + n*[0] + [1]] + [[1]*(n+2)]]), 0, 3)
    for n in [15, 19]
}


def create_fresh_board(n: int):
    fresh_board = 2 * [[[0] * (n + 2)] * (n + 2)] + [[[1] * (n + 2)] + n * [[1] + n * [0] + [1]] + [[1] * (n + 2)]]
    return np.rollaxis(np.array(fresh_board), 0, 3)


class Board:
    """
    Bounded Gomoku Board.
    The boundary stones in the third channel are there to support the learning process
    """

    def __init__(self, n, stones: str = None):

        self.n = n

        class Stone:
            """
            Stones only make sense in the context of a board instance!
            """

            n = self.n

            def __init__(self, r_x: Union[int, str], c_y: int = None):
                """
                We allow matrix coordinates between (1, 1) = left upper and (n, n) = right lower
                or board coordinates where the first argument is a string representing an uppercase letter.
                :param r_x: Row or x coordinate, or ordinal
                :param c_y: Col or y coordinate, or None ir r_x is ordinal
                """
                if c_y is None:
                    assert isinstance(r_x, int), "A single argument must be the ordinal representation"
                    assert self.n + 1 <= r_x <= self.n * self.n, \
                        f"Expecting a number between {self.n+1} and {self.n*self.n}"
                    r, c = divmod(r_x, self.n)
                    x, y = None, None

                elif isinstance(r_x, str):
                    r, c, x, y = None, None, r_x, c_y

                else:
                    x, y, r, c = None, None, r_x, c_y

                if r is not None and c is not None:
                    assert x is None and y is None, \
                        "Please provide either r,c matrix coordinates or x, y, n board coordinates"
                    assert self.n >= r > 0, "row out of range 1 <= r <= n"
                    assert self.n >= c > 0, "col out of range 1 <= c <= n"

                    self.r, self.c = r, c
                if x is not None and y is not None and n is not None:
                    assert r is None and c is None, \
                        "Please provide either r,c matrix coordinates or x, y, n board coordinates"
                    if isinstance(x, str):
                        x = ord(x) - 63
                    self.n, self.r, self.c = n, n-y+1, x-1

                # single-digit representation for vector operations in the ML context
                self.i = self.r * self.n + self.c

            def __str__(self):
                return f"{chr(self.c+64)}{self.n-self.r+1}"

            __repr__ = __str__

            def __eq__(self, other):
                if not isinstance(other, Stone):
                    return False
                elif self.r == other.r and self.c == other.c and self.n == other.n:
                    return True
                else:
                    return False

            def __hash__(self):
                return int(self.i)

            # End of class Stone =============================================

        self.Stone = Stone

        """Set up initial board configuration."""
        # The mathematical representation of the board as (n+2) x (n+2) x 3 tensor
        # Use preconstructed standard fields for performance reasons
        self.math_rep = EMPTY_BOARDS[n].copy() if n in [15, 19] else create_fresh_board(n)

        # Set the stones/bits on the math. representation
        # whoever made the last move is now on channel = 1
        channel = 1
        if isinstance(stones, str):
            stones = self._string_to_stones(stones)

        self._assert_valid(stones)
        self.stones = stones

        if stones is not None:
            for stone in stones[::-1]:
                self.math_rep[stone.r + 1, stone.c + 1, channel] = 1
                channel = 1 - channel

    def _string_to_stones(self, encoded):
        """
        returns an array of pairs for a string-encoded sequence
        e.g. [('A',1), ('M',14)] for 'a1m14'
        """
        x, y = encoded[0].upper(), 0
        stones = []
        for c in encoded[1:]:
            if c.isdigit():
                y = 10 * y + int(c)
            else:
                try:
                    stones.append(self.Stone(*Gt.b2m((x, y), self.n)))
                except AssertionError:
                    raise AssertionError(f"{x}{y} is not a valid position on this board.")
                x = c.upper()
                y = 0
        try:
            stones.append(self.Stone(*Gt.b2m((x, y), self.n)))
        except AssertionError:
            raise AssertionError(f"({x},{y}) is not a valid position on this board.")

        return stones

    def _assert_valid(self, stones: List):
        assert len(set(stones)) == len(stones), f"Stones are not unique: {stones}"
        assert all([self.n > s.r >= 0 for s in stones]), "Not all stones in valid range"
        assert all([self.n > s.c >= 0 for s in stones]), "Not all stones in valid range"

    def plot(self, x_is_next=False):
        def ch(index):
            return [' . ', ' 0 ', ' X ', '   '][index]

        def row(r):
            return f"{self.n-r:2}" if r in range(self.n) else "  "

        rep = self.math_rep
        if x_is_next and len(self.stones) % 2 == 1:
            rep = rep.copy()
            rep[:, :, [0, 1]] = rep[:, :, [1, 0]]

        array = sum([np.rollaxis(rep, -1, 0)[i] * (i+1) for i in range(3)])
        print(f"\n".join([f"{row(i-1)}" + "".join([ch(c) for c in r]) for i, r in enumerate(array)]))
        print("      " + "  ".join([chr(i+65) for i in range(self.n)]))

    def __str__(self):
        return "".join([chr(64+s[0])+str(s[1]) for s in self.stones])

    __repr__ = __str__

    def get_legal_moves(self):
        """
        Returns all the legal moves for the current player
        """
        positions = np.argwhere(np.sum(self.math_rep, axis=-1) == 0)
        return [r * self.n + c for r, c in positions]

    def has_legal_moves(self):
        """Returns True if has legal move else False
        """
        return len(self.get_legal_moves()) > 0

    def put(self, *args):
        if isinstance(args[0], self.Stone):
            stone = args[0]
        else:
            stone = self.Stone(*args)
        m = self.math_rep
        assert m[stone.r, stone.c, 0] == m[stone.r, stone.c, 1] == 0, f"{stone} is occupied."
        m[stone.r, stone.c, 0] = 1
        m[:, :, [0, 1]] = m[:, :, [1, 0]]
        self.stones.append(stone)
        return self


# convenience for playing on the console

A = 'A'
B = 'B'
C = 'C'
D = 'D'
E = 'E'
F = 'F'
G = 'G'
H = 'H'
I = 'I'  # noqa
J = 'J'
K = 'K'
L = 'L'
M = 'M'
N = 'N'
O = 'O'  # noqa
P = 'P'
Q = 'Q'
R = 'R'
S = 'S'

