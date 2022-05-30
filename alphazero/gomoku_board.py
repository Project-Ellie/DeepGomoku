import numpy as np
from domoku.tools import GomokuTools as Gt


class Board:
    def __init__(self, n, stones=None):
        """Set up initial board configuration."""
        self.n = n
        # Create the empty board array.
        self.pieces = create_nxnx2_with_border(n, stones)

    # add [][] indexer syntax to the Board
    def __getitem__(self, index):
        raise NotImplementedError("You may want to implement __getitem__?")
        #  return self.pieces[index]

    def get_legal_moves(self):
        """Returns all the legal moves for the current player
        (1 for white, -1 for black
        """
        moves = set()  # stores the legal moves.

        # Get all empty locations of the current player's plane
        for y in range(self.n):
            for x in range(self.n):
                if self.pieces[x][y][0] == 0:
                    moves.add((x, y))
        return list(moves)

    def has_legal_moves(self):
        """Returns True if has legal move else False
        """
        # Get all empty locations.
        for y in range(self.n):
            for x in range(self.n):
                if self[x][y][0] == 0:
                    return True
        return False

    def execute_move(self, move):
        """Perform the given move on the board; swaps planes
        """
        (x, y) = move
        assert self[x][y] == 0
        self[x][y] = color


def create_nxnx2_with_border(size: int, stones=None):
    """
    Creates a NxNx4 NDArray from the stones of the board. Black is in the 0-plane
    :param size: The side length of the square board
    :param stones: a representation of the current stones on the board: Either string or array of board coordinates
    """
    if isinstance(stones, str):
        stones = Gt.string_to_stones(stones)

    stones = [] if stones is None else stones

    n = size
    board = np.zeros([2, n, n], dtype=np.uint8)

    current = len(stones) % 2
    for move in stones:
        r, c = Gt.b2m(move, n)
        board[current][r][c] = 1
        current = 1 - current

    # next moving player is always on layer 0
    current_layer = np.hstack([
        np.zeros([n + 2, 1], dtype=np.uint8),
        np.vstack([np.zeros([1, n], dtype=np.uint8),
                   board[0],
                   np.zeros([1, n], dtype=np.uint8)]),
        np.zeros([n + 2, 1], dtype=np.uint8)
    ])

    other_layer = np.hstack([
        np.zeros([n + 2, 1], dtype=np.uint8),
        np.vstack([np.zeros([1, n], dtype=np.uint8),
                   board[1],
                   np.zeros([1, n], dtype=np.uint8)]),
        np.zeros([n + 2, 1], dtype=np.uint8)
    ])

    a_border = (size + 2) * [1]
    other_layer[0] = a_border
    other_layer[size + 1] = a_border
    other_layer[:, 0] = a_border
    other_layer[:, size + 1] = a_border

    both = np.array([current_layer, other_layer])
    board = np.rollaxis(both, 0, 3)

    return board
