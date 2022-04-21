import unittest

from domoku.analyzers import Analyzer
from domoku.heuristics import Heuristics
from domoku.board import GomokuBoard
from domoku.constants import *


class MyTestCase(unittest.TestCase):

    def setUp(self) -> None:
        self.heuristics = Heuristics(kappa=3.0)
        self.board_size = 6

    def test_diag_5_detected(self):
        # We have a white updiag in the lower right
        # and a black downdiag in the upper right
        #
        # 6  . B . . . .
        # 5  . . B . . W
        # 4  . . . B W .
        # 3  . . . W B .
        # 2  . . W . . B
        # 1  . W . . . .
        #    A B C D E F
        moves = [[B, 1], [D, 4], [C, 2], [C, 5], [D, 3],
                 [E, 3], [E, 4], [F, 2], [F, 5], [B, 6]]
        self.board = GomokuBoard(self.heuristics, n=self.board_size, disp_width=4)
        for move in moves:
            self.board.set(move[0], move[1])

        result = Analyzer(self.board_size).detect_five(self.board)

        upper = 0
        lower = 1
        right = 1
        # The filters as they are used in the Analyzer
        white_up, white_down, black_up, black_down = 0, 1, 2, 3

        self.assertTrue(result[upper][right][black_down])
        self.assertTrue(result[lower][right][white_up])


if __name__ == '__main__':
    unittest.main()
