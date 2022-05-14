import unittest

from domoku.data import create_binary_rep
from domoku.heuristics import Heuristics
from domoku.board import GomokuBoard
from domoku.constants import *


class MyTestCase(unittest.TestCase):

    def setUp(self) -> None:
        self.heuristics = Heuristics(kappa=3.0)
        self.board_size = 6

    def test_padding(self):
        moves = [[A, 1], [F, 6], [A, 6], [F, 1]]
        self.board = GomokuBoard(n=self.board_size, disp_width=4)
        for move in moves:
            self.board.set(move[0], move[1])

        sample = create_binary_rep(self.board)
        self.assertEqual(sample.shape, (6, 6, 2))

        sample = create_binary_rep(self.board, pad_l=2, pad_r=2, pad_b=2, pad_t=2)
        self.assertEqual(sample.shape, (10, 10, 2))


if __name__ == '__main__':
    unittest.main()
