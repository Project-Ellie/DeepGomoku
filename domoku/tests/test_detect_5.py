import unittest

from domoku.analyzers import Analyzer
from domoku.heuristics import Heuristics
from domoku.board import GomokuBoard
from domoku.constants import *


class MyTestCase(unittest.TestCase):

    def setUp(self) -> None:
        self.board_size = 6

    def test_diag_5_detected(self):
        # We have a white updiag in the lower right
        # and a black downdiag in the upper right
        #
        # 6  . B . . . .
        # 5  . B B . . W
        # 4  . B . B W .
        # 3  . B . W B .
        # 2  . B W . . B
        # 1  W W W W W .
        #    A B C D E F
        moves = [[B, 1], [D, 4], [C, 2], [C, 5], [D, 3],
                 [E, 3], [E, 4], [F, 2], [F, 5], [B, 6],
                 [A, 1], [B, 5], [C, 1], [B, 4], [D, 1], [B, 3], [E, 1], [B, 2], ]
        self.board = GomokuBoard(n=self.board_size, disp_width=4)
        for move in moves:
            self.board.set(move[0], move[1])

        result = Analyzer(self.board_size).detect_five(self.board)

        # The filters as they are used in the Analyzer
        white_up, white_down, white_hor, white_ver = 0, 1, 2, 3
        black_up, black_down, black_hor, black_ver = 4, 5, 6, 7

        # The coordinates are the array coordinates of the pattern centers
        self.assertTrue(result[2][3][black_down])
        self.assertTrue(result[3][3][white_up])
        self.assertTrue(result[2][1][black_ver])
        self.assertTrue(result[5][2][white_hor])


if __name__ == '__main__':
    unittest.main()
