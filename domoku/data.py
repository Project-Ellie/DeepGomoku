import numpy as np
from domoku.tools import GomokuTools as gtools


def transform(stones, n, quarters, reflect=False):
    """
    return stones' coordinates after rotation and reflection
    """
    coords = [gtools.b2m(stone, n) for stone in stones]
    if quarters == 0:
        res = coords

    elif quarters == 1:
        res = [(n - c - 1, r) for r, c in coords]

    elif quarters == 2:
        res = [(n - r - 1, n - c - 1) for r, c in coords]

    elif quarters == 3:
        res = [(c, n - r - 1) for r, c in coords]

    else:
        raise ValueError("quaters can only be 0, 1, 2, or 3")

    if reflect:
        res = [(r, n - c - 1) for r, c in res]
        
    stones = [gtools.m2b(coord, n) for coord in res]
    return stones


def create_binary_action(board_size, padding, position, switch=False):
    x, y = np.array([padding, padding]) + gtools.b2m(position, board_size)
    size = 2 * padding + board_size
    action = np.zeros([size, size, 2])
    layer = 1 if switch else 0
    action[x, y, layer] = 1
    return action


def create_binary_rep(n, stones, current_color, pad_r=0, pad_l=0, pad_t=0, pad_b=0,
                      padding=None, border=False, switch=False):
    """
    Creates a NxNx2 NDArray from the stones of the board. Black is in the 0-plane
    """
    pad_r = pad_r if padding is None else padding
    pad_l = pad_l if padding is None else padding
    pad_t = pad_t if padding is None else padding
    pad_b = pad_b if padding is None else padding

    assert not (padding is None and border), "must have padding > 0 when requesting border"

    sample = np.zeros([2, n, n], dtype=np.uint8)

    current = current_color
    if switch:
        current = 1 - current
    for move in stones:
        r, c = gtools.b2m(move, n)
        sample[current][r][c] = 1
        current = 1 - current

    # next moving player is always on layer 0
    current_layer = np.hstack([
        np.zeros([n + pad_l + pad_r, pad_t], dtype=np.uint8),
        np.vstack([np.zeros([pad_l, n], dtype=np.uint8),
                   sample[0],
                   np.zeros([pad_r, n], dtype=np.uint8)]),
        np.zeros([n + pad_l + pad_r, pad_b], dtype=np.uint8)
    ])

    other_layer = np.hstack([
        np.zeros([n + pad_l + pad_r, pad_t], dtype=np.uint8),
        np.vstack([np.zeros([pad_l, n], dtype=np.uint8),
                   sample[1],
                   np.zeros([pad_r, n], dtype=np.uint8)]),
        np.zeros([n + pad_l + pad_r, pad_b], dtype=np.uint8)
    ])

    if border:
        size = n
        a_border = (padding - 1) * [0] + (size + 2) * [1] + (padding - 1) * [0]
        other_layer[padding - 1] = a_border
        other_layer[size + padding] = a_border
        other_layer[:, padding - 1] = a_border
        other_layer[:, size + padding] = a_border

    both = np.array([current_layer, other_layer])

    return np.rollaxis(both, 0, 3).astype(float)


def create_nxnx4(size: int, stones=None, pad_r=0, pad_l=0, pad_t=0, pad_b=0,
                 padding=None, border=False, switch=False):
    """
    Creates a NxNx4 NDArray from the stones of the board. Black is in the 0-plane
    """
    if isinstance(stones, str):
        stones = gtools.string_to_stones(stones)

    stones = [] if stones is None else stones
    pad_r = pad_r if padding is None else padding
    pad_l = pad_l if padding is None else padding
    pad_t = pad_t if padding is None else padding
    pad_b = pad_b if padding is None else padding

    assert not (padding is None and border), "must have padding > 0 when requesting border"

    n = size
    sample = np.zeros([2, n, n], dtype=np.uint8)

    current = len(stones) % 2
    if switch:
        current = 1 - current
    for move in stones:
        r, c = gtools.b2m(move, n)
        sample[current][r][c] = 1
        current = 1 - current

    # next moving player is always on layer 0
    current_layer = np.hstack([
        np.zeros([n + pad_l + pad_r, pad_t], dtype=np.uint8),
        np.vstack([np.zeros([pad_l, n], dtype=np.uint8),
                   sample[0],
                   np.zeros([pad_r, n], dtype=np.uint8)]),
        np.zeros([n + pad_l + pad_r, pad_b], dtype=np.uint8)
    ])

    other_layer = np.hstack([
        np.zeros([n + pad_l + pad_r, pad_t], dtype=np.uint8),
        np.vstack([np.zeros([pad_l, n], dtype=np.uint8),
                   sample[1],
                   np.zeros([pad_r, n], dtype=np.uint8)]),
        np.zeros([n + pad_l + pad_r, pad_b], dtype=np.uint8)
    ])

    if border:
        a_border = (padding - 1) * [0] + (size + 2) * [1] + (padding - 1) * [0]
        other_layer[padding - 1] = a_border
        other_layer[size + padding] = a_border
        other_layer[:, padding - 1] = a_border
        other_layer[:, size + padding] = a_border

    both = np.array([current_layer, other_layer])
    sample = np.rollaxis(both, 0, 3).astype(float)
    sample = np.stack([sample, np.zeros((size, size, 2))], axis=2).reshape((size, size, 4))

    return sample


def get_winning_color(sample, winning_channel):
    """
    :param sample: A any nxnx4 board numpy board representation
    :param winning_channel: the channel containing the winning pattern, usually a line of 5
    :return: 0 (black) if the first player owns the winning channel, else 1 (white)
    """
    if winning_channel is None:
        return None
    current_player_color = np.sum(sample, axis=None).astype(int) % 2
    winning_color = (current_player_color + winning_channel) % 2
    return winning_color


def after(sample, move):
    """
    Basically a move in matrix coordinates.
    :param sample: a nxnx4 np array rep of the board
    :param move: a move in matrix coords - NOT board coords!!
    :return: A new sample, with that stone, current and other players' channels reversed
    """
    sample = sample.copy()
    row, column = move
    row = int(row)
    column = int(column)
    try:
        sample[row][column][0] = 1
    except IndexError as ie:
        print("oops")
        raise ie

    n = sample.shape[0]
    zeros = np.zeros((n, n))
    new_board = np.rollaxis(np.stack([sample[:, :, 1], sample[:, :, 0], zeros, zeros]), 0, 3)
    return new_board
