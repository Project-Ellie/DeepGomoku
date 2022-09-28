import abc
import copy
from typing import Tuple, List, Any

import numpy as np
from numpy import ndarray

from aegomoku.gomoku_board import stones_from_example
from aegomoku.interfaces import Adviser, Board


def _get_key(state):
    field, phase = state
    example = field, [], 0
    stones, _ = stones_from_example(example)
    res = str(sorted(stones))
    return res + str(phase)


class TestAdviser(Adviser):
    """
    Advises along a given fixed trajectory beginning with the given board
    """
    def __init__(self, board: Board, moves: List[Any]):
        """
        :param board: an initial board
        :param moves: the trajectory along which to advise
        """
        board = copy.deepcopy(board)
        self.board = board
        self.moves = {}
        for i in range(len(moves)):
            key = _get_key(board.canonical_representation())
            self.moves[key] = moves[i]
            board.act(moves[i])

    @abc.abstractmethod
    def get_advisable_actions(self, state: Tuple[ndarray, List[int]]) -> List[float]:
        key = _get_key(state)
        return [self.moves.get(key)]

    @abc.abstractmethod
    def advise(self, state: Tuple[ndarray, List[int]]):
        """
        :param state: A pair of board rep NxNx3 and the one-hot encoded phase

        :returns:
            pi: a policy vector for the current board- a numpy array of length
                game.get_action_size
            v: a float in [-1,1] that represents the value of the current board
        """
        probs = np.zeros(shape=(len(self.board.get_legal_actions()) + 1))
        probs[self.get_advisable_actions(state)[0]] = 1.0
        return probs
