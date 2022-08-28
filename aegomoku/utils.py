import numpy as np
from timeit import default_timer
import aegomoku.tools as gt
from aegomoku.gomoku_board import GomokuBoard
from aegomoku.mpl_board import MplBoard


def analyse_board(board_size, stones, policy, suppress_move_numbers=False, disp_width=6):
    if all([isinstance(i, (np.integer, int)) for i in stones]):
        stones = [gt.m2b2(divmod(i, 15), 15) for i in stones]
    lb = MplBoard(n=board_size, disp_width=disp_width, stones=stones, heuristics=policy,
                  suppress_move_numbers=suppress_move_numbers)
    lb.display()


def analyse_example(board_size, example, disp_width=6):

    s, p, v = example
    n_current = np.sum(s[:, :, 0], axis=None)
    n_other = np.sum(s[:, :, 1], axis=None)
    if n_other == n_current:
        current = 'BLACK'
        black = 0
    else:
        current = 'WHITE'
        black = 1
    print(f"Next to play: {current}")
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
    analyse_board(board_size, stones, policy=p, suppress_move_numbers=True, disp_width=disp_width)
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
