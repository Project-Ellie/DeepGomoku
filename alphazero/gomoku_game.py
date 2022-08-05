import copy
from typing import Tuple

import random
import abc

import numpy as np
from alphazero.interfaces import Game, Move
from alphazero.gomoku_board import GomokuBoard
from alphazero.policies.heuristic_policy import HeuristicPolicy


# Remove this
def __initial_stones(board_size: int, n_stones: int):
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


class BoardInitializer:
    @abc.abstractmethod
    def initial_stones(self):
        pass


class ConstantBoardInitializer(BoardInitializer):

    def __init__(self, stones: str):
        self.stones = stones

    def initial_stones(self):
        return self.stones


class RandomBoardInitializer(BoardInitializer):

    def __init__(self, board_size, num_stones, left=0, right=None, upper=0, lower=None):
        self.right = right if right is not None else board_size - 1
        self.lower = lower if lower is not None else board_size - 1
        self.left = left
        self.upper = upper
        self.num_stones = num_stones
        self.board_size = board_size

    def initial_stones(self):
        stones = ""
        n_it = 0
        for _ in range(self.num_stones):
            while True and n_it < self.board_size ** 2:
                col = random.randint(self.left, self.right)
                row = random.randint(self.upper, self.lower)
                stone = chr(col + 65) + str(self.board_size-row)
                if stone not in stones:
                    stones += stone
                    break
            if n_it == self.board_size ** 2:
                raise ValueError(f"Tried {self.board_size ** 2} times but failed.")
        return stones


class GomokuGame(Game):
    def __init__(self, board_size, initializer: BoardInitializer = None):
        super().__init__()
        self.board_size = board_size
        self.initializer = initializer
        self.n_in_row = 5
        self.detector = None

    def get_initial_board(self) -> GomokuBoard:
        initial_stones = self.initializer.initial_stones()
        return GomokuBoard(self.board_size, stones=initial_stones)

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
        if self.detector is None:
            self.detector = HeuristicPolicy(self.board_size)
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
