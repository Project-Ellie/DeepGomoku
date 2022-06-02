from unittest import TestCase

from alphazero.interfaces import IllegalMoveException
from gomoku_board import Board, Move


class GomokuBoardTests(TestCase):

    def test_illegal_moves_rejected(self):
        self.assertRaises(IllegalMoveException, lambda: Board(9, 'd4d4'))

