from typing import List, Tuple

import numpy as np
from timeit import default_timer
import aegomoku.tools as gt
from aegomoku.gomoku_board import GomokuBoard
from aegomoku.mpl_board import MplBoard


def analyse_board(board_size, stones, adviser_or_array,
                  suppress_move_numbers=False, disp_width: float = 6, policy_cutoff: float = 1e-5):
    if all([isinstance(i, (np.integer, int)) for i in stones]):
        b2 = []
        for i in stones:
            stone = gt.m2b2(divmod(i, board_size), board_size)
            b2.append(stone)
    else:
        b2 = stones

    lb = MplBoard(n=board_size, disp_width=disp_width, stones=b2, adviser=adviser_or_array,
                  suppress_move_numbers=suppress_move_numbers, policy_cutoff=policy_cutoff)
    lb.display()


def stones_from_example(example) -> Tuple[List[int], str]:

    s, _, _ = example
    s = np.squeeze(s)
    board_size = s.shape[0] - 2
    n_current = np.sum(s[:, :, 0], axis=None)
    n_other = np.sum(s[:, :, 1], axis=None)
    if n_other == n_current:
        current = 'BLACK'
        black = 0
    else:
        current = 'WHITE'
        black = 1
    whites = np.where(s[:, :, 1 - black] == 1)
    blacks = np.where(s[:, :, black] == 1)
    whites = list((whites[0] - 1) * board_size + whites[1]-1)
    blacks = list((blacks[0] - 1) * board_size + blacks[1]-1)
    stones = []
    while True:
        try:
            stones.append(int(blacks.pop()))
            stones.append(int(whites.pop()))
        except IndexError:
            break
    return stones, current


def analyse_example(example, disp_width=7.5, policy_cutoff=1e-5):
    s, p, v = example
    board_size = int(np.sqrt(len(p)))
    stones, current = stones_from_example(example)
    analyse_board(board_size, stones, adviser_or_array=p, suppress_move_numbers=True, disp_width=disp_width,
                  policy_cutoff=policy_cutoff)
    print(f"Next to play: {current}")
    print(f"Value from {current}'s point of view: {v}")


def expand(the_board):
    """
    Expand the NxNx3 representation of the board to prepare for ingestion into neural networks
    :param the_board: either NxNx3 or a GomokuBoard instance
    :return:
    """
    if isinstance(the_board, GomokuBoard):
        state = the_board.math_rep
    else:
        state = the_board
    return np.expand_dims(state, axis=0).astype(float)


class AverageMeter(object):
    """From https://github.com/pytorch/examples/blob/master/imagenet/main.py"""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def __repr__(self):
        return f'{self.avg:.2e}'

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class DotDict(dict):
    def __getattr__(self, name):
        return self[name]


class Timer(object):
    def __init__(self, verbose=False):
        self.verbose = verbose
        self.timer = default_timer


    def __enter__(self):
        self.start = self.timer()
        return self


    def __exit__(self, *args):
        end = self.timer()
        self.elapsed_secs = end - self.start
        self.elapsed = self.elapsed_secs * 1000  # millisecs
        if self.verbose:
            print('elapsed time: %f ms' % self.elapsed)
