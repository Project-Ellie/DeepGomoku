import numpy as np


def ver_1xnxn(pattern):
    n = len(pattern)
    lp = n // 2
    return np.vstack([np.zeros((lp, n)),
                      np.reshape(pattern, (1, n)),
                      np.zeros((lp, n))])


def hor_1xnxn(pattern):
    n = len(pattern)
    lp = n // 2
    return np.hstack([np.zeros((n, lp)),
                      np.reshape(pattern, (n, 1)),
                      np.zeros((n, lp))])


def all_2xnxn(pattern):
    current, other = pattern
    raw_stack = np.stack([
        np.stack([hor_1xnxn(current), hor_1xnxn(other)]),
        np.stack([np.diag(current), np.diag(other)]),
        np.stack([ver_1xnxn(current), ver_1xnxn(other)]),
        np.stack([np.diag(current)[::-1], np.diag(other)[::-1]]),
    ])
    return np.rollaxis(np.rollaxis(raw_stack, 1, 4), 0, 4)


def radial_2xnxn(pattern_curr, pattern_oth=None, center_curr=-1, center_oth=-1, gamma=1.0):
    pattern_oth = pattern_oth if pattern_oth is not None else pattern_curr
    assert pattern_oth is None or len(pattern_oth) == len(pattern_curr), 'Patterns must have same length'
    lp = len(pattern_curr)

    curr = pattern_curr + [0] + pattern_curr[::-1]
    oth = pattern_oth + [0] + pattern_oth[::-1]

    res = np.sum(all_2xnxn([curr, oth]), axis=-1) * gamma
    res[lp, lp, :] = [center_curr, center_oth]
    return res

