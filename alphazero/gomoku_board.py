from typing import List, Union

import numpy as np


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

    def __init__(self, board_size, stones: str = None, x_means='black'):
        """
        :param board_size: Usable side length without boundary perimeter
        :param stones: current stones on the board as a single string like 'h8f7g12d8' or so.
        :param x_means: for plotting: 'black' for X  always black or 'next' for X always next
        """
        assert isinstance(x_means, str) and len(x_means) > 0 and x_means[0] in ['B', 'b', 'N', 'n'], \
            "x_means must be a string beginning with 'B' or 'b' for black or 'N' or 'n' for next"

        self.x_is_next = True if x_means[0] in ['N', 'n'] else False if x_means[0] in ['B', 'b'] else None
        self.board_size = board_size

        class Stone:
            """
            Stones only make sense in the context of a board instance!
            """

            # Note that this is Board.self!
            board_size = self.board_size
            field_size = self.board_size + 2  # The field includes the border

            ord_min = field_size + 1
            ord_max = field_size * (field_size - 1) - 2

            all_actions = set([(board_size+2) * r + c + 1 for r in range(1, board_size) for c in range(board_size)])

            def __init__(self, r_x: Union[int, str], c_y: int = None):
                """
                We allow matrix coordinates between (1, 1) = left upper and (n, n) = right lower
                or board coordinates where the first argument is a string representing an uppercase letter.
                :param r_x: Row or x coordinate, or ordinal
                :param c_y: Col or y coordinate, or None ir r_x is ordinal
                """
                n = self.board_size

                if c_y is None:
                    if isinstance(r_x, int):
                        assert self.ord_min <= r_x <= self.ord_max, \
                            f"Expecting a number between {self.ord_min} and {self.ord_max}"
                        r, c = divmod(r_x, self.field_size)
                        x, y = None, None
                    elif isinstance(r_x, str):
                        x, y = r_x[0], int(r_x[1:])
                        r, c = None, None
                    else:
                        raise ValueError("If a single argument is provided, "
                                         "it must be a string or integer representation of the move.")

                elif isinstance(r_x, str):
                    r, c, x, y = None, None, r_x, c_y

                else:
                    x, y, r, c = None, None, r_x, c_y

                if r is not None and c is not None:
                    assert x is None and y is None, \
                        "Please provide either r,c matrix coordinates or x, y, n board coordinates"
                    if c == 10:
                        print("oops")
                    assert n > r >= 0, f"row {r} out of range 0 <= r < {n}"
                    assert n > c >= 0, f"col {c} out of range 0 <= c < {n}"

                    self.r, self.c = r, c
                if x is not None and y is not None and n is not None:
                    assert r is None and c is None, \
                        "Please provide either r,c matrix coordinates or x, y, n board coordinates"
                    if isinstance(x, str):
                        x = ord(x) - 64
                    self.r, self.c = n-y+1, x

                # single-digit representation for vector operations in the ML context
                self.i = self.r * self.field_size + self.c

            def __str__(self):
                return f"{chr(self.c+64)}{self.field_size-self.r-1}"

            __repr__ = __str__

            def __eq__(self, other):
                if not isinstance(other, Stone):
                    return False
                elif self.r == other.r and self.c == other.c and self.field_size == other.field_size:
                    return True
                else:
                    return False

            def __hash__(self):
                return int(self.i)

            # End of class Stone =============================================

        self.Stone = Stone

        """Set up initial board configuration."""
        # The mathematical representation of the board as (n+2) x (n+2) x 3 tensor
        # Use preconstructed standard fields for n=15, 19 for performance improvement.
        size = self.board_size
        self.math_rep = EMPTY_BOARDS[size].copy() if size in [15, 19] else create_fresh_board(size)

        # Set the stones/bits on the math. representation
        # whoever made the last move is now on channel = 1
        channel = 1
        if isinstance(stones, str):
            stones = self._string_to_stones(stones)

        self.stones = stones if stones is not None else ""

        if stones is not None:
            self._assert_valid(stones)
            for stone in stones[::-1]:
                self.math_rep[stone.r, stone.c, channel] = 1
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
                    stones.append(self.Stone(x, y))
                except AssertionError:
                    raise AssertionError(f"{x}{y} is not a valid position on this board.")
                x = c.upper()
                y = 0
        try:
            stones.append(self.Stone(x, y))
        except AssertionError:
            raise AssertionError(f"({x},{y}) is not a valid position on this board.")

        return stones

    def _assert_valid(self, stones: List):
        assert len(set(stones)) == len(stones), f"Stones are not unique: {stones}"
        assert all([self.board_size >= s.r >= 1 for s in stones]), "Not all stones in valid range"
        assert all([self.board_size >= s.c >= 1 for s in stones]), "Not all stones in valid range"

    def plot(self, x_is_next=None):

        x_is_next = x_is_next if x_is_next is not None else self.x_is_next

        def ch(index):
            return [' . ', ' X ', ' 0 ', '   '][index]

        def row(r):
            return f"{self.board_size-r:2}" if r in range(self.board_size) else "  "

        rep = self.math_rep
        if not x_is_next and len(self.stones) % 2 == 1:
            rep = rep.copy()
            rep[:, :, [0, 1]] = rep[:, :, [1, 0]]

        array = sum([np.rollaxis(rep, -1, 0)[i] * (i+1) for i in range(3)])
        print(f"\n".join([f"{row(i-1)}" + "".join([ch(c) for c in r]) for i, r in enumerate(array)]))
        print("      " + "  ".join([chr(i+65) for i in range(self.board_size)]))

    def __str__(self):
        return "".join(f"{s} " for s in self.stones)

    __repr__ = __str__

    def get_legal_actions(self):
        """
        Returns all the legal moves for the current player
        """
        positions = np.argwhere(np.sum(self.math_rep, axis=-1) == 0)
        return [r * (self.board_size + 2) + c for r, c in positions]

    def get_legal_moves(self):
        actions = self.get_legal_actions()
        return [self.Stone(action) for action in actions]

    def has_legal_moves(self):
        """Returns True if has legal move else False
        """
        return len(self.get_legal_actions()) > 0

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
