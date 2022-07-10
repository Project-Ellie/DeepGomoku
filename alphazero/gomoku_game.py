import copy
from typing import Tuple, Callable, Union

import numpy as np
from alphazero.interfaces import Game, TerminalDetector, Move
from alphazero.gomoku_board import GomokuBoard


def initial_stones(board_size: int, n_stones: int):
    """
    :return: a function that returns a list of n_stones randomly positioned stones,
        at least 3 positions away from the boundary
    """
    def inner():
        move = GomokuBoard(board_size).Stone
        the_stones = set()
        while len(the_stones) < n_stones:
            the_stones.add(move(np.random.randint(3, 12), np.random.randint(3, board_size-3)))
        return list(the_stones)
    return inner


class GomokuGame(Game):
    def __init__(self, board_size, detector: TerminalDetector, initial: Union[str, Callable] = None):
        super().__init__()
        self.board_size = board_size
        self.initial_stones = initial if initial is not None else ""
        self.n_in_row = 5
        self.detector = detector

    def get_initial_board(self) -> GomokuBoard:
        if isinstance(self.initial_stones, str):
            return GomokuBoard(self.board_size, stones=self.initial_stones)
        elif isinstance(self.initial_stones, Callable):
            return GomokuBoard(self.board_size, stones=self.initial_stones())

    def get_board_size(self, board) -> int:
        return self.board_size ** 2

    def get_action_size(self, board: GomokuBoard):
        # return number of actions
        return self.board_size ** 2

    def get_next_state(self, board: GomokuBoard, action: Move) -> Tuple[GomokuBoard, int]:
        """
        computes the next state from a deep copy. Leaves the passed board unchanged
        :return:
        """
        board = copy.deepcopy(board)
        board.act(action)
        next_player = board.get_current_player()
        return board, next_player

    def get_valid_moves(self, board: GomokuBoard):
        bits = np.zeros([self.board_size * self.board_size])
        legal_indices = board.get_legal_actions()
        bits[legal_indices] = 1
        return bits

    def get_game_ended(self, board: GomokuBoard):
        return self.detector.get_winner(board.canonical_representation())

    # modified
    def get_symmetries(self, board, pi):
        """
        :param board: np array with the stones
        :param pi: np array containing the move probabilities for the current player
        :return:
        """
        # TODO: Implement this function according to spec

        # mirror, rotational
        n = self.board_size
        assert np.shape(pi) == (n * n,) or np.shape(pi) == (n, n), "pi should be square or flattened."
        if np.shape(pi) == (n * n,):
            pi_board = np.reshape(pi, (self.board_size, self.board_size))
        else:
            pi_board = pi
        symmetries = []

        for i in range(1, 5):
            for j in [True, False]:
                new_b = np.rot90(board, i, axes=(0, 1))
                new_pi = np.rot90(pi_board, i, axes=(0, 1))
                if j:
                    new_b = np.fliplr(new_b)
                    new_pi = np.fliplr(new_pi)
                symmetries += [(new_b, list(new_pi.ravel()))]
        return symmetries
