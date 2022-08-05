from timeit import default_timer
from alphazero.mpl_board import MplBoard


def analyse_board(board_size, stones, policy):
    stones_str = "".join([str(stone) for stone in stones])
    lb = MplBoard(n=board_size, disp_width=8, stones=stones_str, heuristics=policy)
    lb.display()


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
