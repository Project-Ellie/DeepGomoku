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


def all_3xnxn(pattern):
    current, other, boundary = pattern
    raw_stack = np.stack([
        np.stack([hor_1xnxn(current), hor_1xnxn(other), hor_1xnxn(boundary)]),
        np.stack([np.diag(current), np.diag(other), np.diag(boundary)]),
        np.stack([ver_1xnxn(current), ver_1xnxn(other), ver_1xnxn(boundary)]),
        np.stack([np.diag(current)[::-1], np.diag(other)[::-1], np.diag(boundary)[::-1]]),
    ])
    return np.rollaxis(np.rollaxis(raw_stack, 1, 4), 0, 4)


def all_5xnxn(pattern, pov):
    """
    compute the 9x9x5x4 filters for secondary patterns
    :param pattern:
    :param pov:
    :return:
    """
    stones, influence, defense = pattern
    size = len(stones)
    dont_care = np.zeros((size, size))
    if pov == 0:
        raw_stack = np.stack([
            #                 c                  o                b                        i                j
            np.stack([hor_1xnxn(stones), hor_1xnxn(defense), hor_1xnxn(defense), hor_1xnxn(influence), dont_care]),
            np.stack([np.diag(stones), np.diag(defense), np.diag(defense), np.diag(influence), dont_care]),
            np.stack([ver_1xnxn(stones), ver_1xnxn(defense), ver_1xnxn(defense), ver_1xnxn(influence), dont_care]),
            np.stack([np.diag(stones)[::-1], np.diag(defense)[::-1], np.diag(defense)[::-1], np.diag(influence)[::-1],
                      dont_care]),
        ])
    else:
        raw_stack = np.stack([
            #                 c                  o                b                     i                j
            np.stack([hor_1xnxn(defense), hor_1xnxn(stones), hor_1xnxn(defense), dont_care, hor_1xnxn(influence)]),
            np.stack([np.diag(defense), np.diag(stones), np.diag(defense), dont_care, np.diag(influence)]),
            np.stack([ver_1xnxn(defense), ver_1xnxn(stones), ver_1xnxn(defense), dont_care, ver_1xnxn(influence)]),
            np.stack([np.diag(defense)[::-1], np.diag(stones)[::-1], np.diag(defense)[::-1], dont_care,
                      np.diag(influence)[::-1]]),
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


def radial_3xnxn(pattern_curr, pattern_oth=None, pattern_bnd=None,
                 center_curr=-1, center_oth=-1, center_bnd=-1, gamma=1.0):
    """
    This is where the defensive character of the other player's and the boundary stones is taken into account
    """

    assert pattern_oth is None or len(pattern_oth) == len(pattern_curr), 'Patterns must have same length'
    assert pattern_bnd is None or len(pattern_bnd) == len(pattern_curr), 'Patterns must have same length'

    pattern_oth = pattern_oth if pattern_oth is not None else pattern_curr
    pattern_bnd = pattern_bnd if pattern_bnd is not None else pattern_curr

    lp = len(pattern_curr)

    curr = pattern_curr + [0] + pattern_curr[::-1]
    oth = pattern_oth + [0] + pattern_oth[::-1]
    bnd = pattern_bnd + [0] + pattern_bnd[::-1]

    # This actually subtracts the defensive influence from the offensive, as defensive patterns and boundary patterns
    # come with a negative sign
    res = np.sum(all_3xnxn([curr, oth, bnd]), axis=-1) * gamma
    res[lp, lp, :] = [center_curr, center_oth, center_bnd]
    return res
