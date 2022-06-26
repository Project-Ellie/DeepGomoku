from typing import List, Union, Callable

import bitarray
import numpy as np
from alphazero.interfaces import Board

EMPTY_BOARDS = {
    n: np.rollaxis(
        np.array(2 * [[[0] * (n + 2)] * (n + 2)] + [[[1]*(n+2)] + n * [[1] + n*[0] + [1]] + [[1]*(n+2)]]), 0, 3)
    for n in [15, 19]
}


def create_fresh_board(n: int):
    fresh_board = 2 * [[[0] * (n + 2)] * (n + 2)] + [[[1] * (n + 2)] + n * [[1] + n * [0] + [1]] + [[1] * (n + 2)]]
    return np.rollaxis(np.array(fresh_board), 0, 3)


class GomokuBoard(Board):
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

            ord_min = 0
            ord_max = board_size * board_size - 1

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
                        r, c = divmod(r_x, self.board_size)
                        x, y = None, None
                    elif isinstance(r_x, str):
                        x, y = r_x[0], int(r_x[1:])
                        r, c = None, None
                    elif isinstance(r_x, List):
                        r_x = int(np.argmax(r_x))
                        assert self.ord_min <= r_x <= self.ord_max, \
                            f"Expecting a number between {self.ord_min} and {self.ord_max}"
                        r, c = divmod(r_x, self.board_size)
                        x, y = None, None
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
                    assert n > r >= 0, f"row {r} out of range 0 <= r < {n}"
                    assert n > c >= 0, f"col {c} out of range 0 <= c < {n}"

                    self.r, self.c = r, c
                if x is not None and y is not None and n is not None:
                    assert r is None and c is None, \
                        "Please provide either r,c matrix coordinates or x, y, n board coordinates"
                    if isinstance(x, str):
                        x = ord(x) - 64
                    self.r, self.c = n-y, x-1

                # single-digit representation for vector operations in the ML context
                self.i = self.r * self.board_size + self.c

            def __str__(self):
                return f"{chr(self.c+65)}{self.board_size-self.r}"

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

            def xy(self):
                """
                X,Y coordinates as on the physical board, A=1, B=2, etc
                :return:
                """
                return self.board_size - self.r, self.c + 1

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
            stones = self.string_to_stones(stones)

        self.stones = stones if stones is not None else []

        if stones is not None and len(stones) > 0:
            self._assert_valid(stones)
            for stone in stones[::-1]:
                self.math_rep[stone.r + 1, stone.c + 1, channel] = 1
                channel = 1 - channel

        else:  # add a tiny polution, because the maths (softmax) don't like zeros only
            self.math_rep[self.board_size // 2 - 1][self.board_size // 2][0] = 1e-5

    def get_current_player(self) -> int:
        return len(self.get_stones()) % 2

    def get_stones(self):
        return self.stones

    def string_to_stones(self, encoded):
        """
        returns an array of Stones for a string-encoded sequence
        e.g. [Stone('A1'), Stone('M14')] for 'a1m14'
        """
        if encoded is None or encoded == '':
            return []

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

    def plot(self, x_is_next=None, mark=None):
        mark = self.stones[-1] if mark is None else mark

        x_is_next = x_is_next if x_is_next is not None else self.x_is_next

        def ch(index, i, j):
            symbol = [' . ', ' X ', ' O ', '   '][index]
            if mark is not None and (i-1, j-1) == (mark.r, mark.c):
                symbol = f'[{symbol.strip()}]'
            return symbol

        def row(r):
            return f"{self.board_size-r:2}" if r in range(self.board_size) else "  "

        rep = self.math_rep
        if not x_is_next and len(self.stones) % 2 == 1:
            rep = rep.copy()
            rep[:, :, [0, 1]] = rep[:, :, [1, 0]]

        array = sum([np.rollaxis(rep, -1, 0)[i] * (i+1) for i in range(3)])
        print(f"\n".join([f"{row(i-1)}" + "".join([ch(c, i, j)
                                                   for j, c in enumerate(r)]) for i, r in enumerate(array)]))
        print("      " + "  ".join([chr(i+65) for i in range(self.board_size)]))


    def print_pi(self, policy: Callable, scale=None):
        """
        Print the policy values as upscaled integers
        :param policy: a policy that takes the math_rep of this board
        :param scale: any factor that makes the output readable
        """
        pi = np.squeeze(policy(self.math_rep)).reshape((self.board_size,  self.board_size))
        mx = np.max(pi, axis=None)
        if mx == 0:
            mx = 1
        scale = scale if scale is not None else 999. / mx
        print((scale*pi).astype(int))


    def __str__(self):
        return " ".join(str(s) for s in self.stones)

    __repr__ = __str__

    def get_string_representation(self):
        """
        :return: hash string of the board bits. Note that we don't use the moves,
         because the order of the moves must not matter for this method.
        """
        bita = bitarray.bitarray(list(self.math_rep.flatten()))
        return str(hash(str(bita)))

    def get_legal_actions(self):
        """
        Returns all the legal moves for the current player
        """
        positions = np.argwhere(np.sum(self.math_rep, axis=-1) == 0) - np.array([1, 1])
        return [r * self.board_size + c for r, c in positions]

    def get_legal_moves(self):
        actions = self.get_legal_actions()
        return [self.Stone(action) for action in actions]

    def has_legal_moves(self):
        """Returns True if has legal move else False
        """
        return len(self.get_legal_actions()) > 0

    def act(self, *args):
        if isinstance(args[0], self.Stone):
            stone = args[0]
        else:
            stone = self.Stone(*args)
        m = self.math_rep
        assert m[stone.r+1, stone.c+1, 0] == m[stone.r+1, stone.c+1, 1] == 0, f"{stone} is occupied."
        m[stone.r+1, stone.c+1, 0] = 1
        self.swap()
        self.stones.append(stone)
        return self

    def undo(self):
        if not self.stones:
            return self
        self.remove(self.stones[-1])
        self.swap()
        return self

    def swap(self):
        self.math_rep[:, :, [0, 1]] = self.math_rep[:, :, [1, 0]]

    def canonical_representation(self):
        return self.math_rep


    def remove(self, stone):
        assert isinstance(stone, self.Stone), "Can only remove this board's stones"
        self.math_rep[stone.r+1, stone.c+1] = [0, 0, 0]
        self.stones.remove(stone)

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
