import numpy as np
from domoku.tools import GomokuTools as gtools
from domoku.board import GomokuBoard
from domoku.qfunction import heuristic_QF

BLACK = 0
WHITE = 1
EDGES = 2
STYLE_MIXED = 2


def roll_out(board, policy, max_n=40):
    """
    takes the board and subsequently applies the given policy to black and white
    until the game is decided or max_n moves have been played.
    """
    n = 0
    board.compute_all_scores()
    move = policy.suggest(board)
    while move.status == 0 and n < max_n:
        board.set(move.x, move.y)
        board.compute_all_scores()
        move = policy.suggest(board)
        n += 1
    return board


def variants_for(board):
    """
    Create a tensor 8x2xNxN to represent the 8 equivalent boards 
    that can be created from the stones by reflection and rotation.
    """
    stones = board.stones.copy()
    n = board.N
    array = np.zeros([8, 2, n, n], dtype=float)
    color = np.arange(len(stones)) % 2
    for ll, pos in list(zip(color, stones)):
        r, c = gtools.b2m(pos, 15)
        array[0][ll][r][c] = 1.0
        array[6][ll][c][r] = 1.0

        array[1][ll][c][n-r] = 1.0
        array[4][ll][n-r][c] = 1.0

        array[3][ll][n-c][r] = 1.0
        array[7][ll][r][n-c] = 1.0

        array[2][ll][n-r][n-c] = 1.0
        array[5][ll][n-c][n-r] = 1.0

    return array


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


def create_binary_rep(board: GomokuBoard, pad_r=0, pad_l=0, pad_t=0, pad_b=0,
                      padding=None, border=False, switch=False):
    """
    Creates a NxNx2 NDArray from the stones of the board. Black is in the 0-plane
    """
    pad_r = pad_r if padding is None else padding
    pad_l = pad_l if padding is None else padding
    pad_t = pad_t if padding is None else padding
    pad_b = pad_b if padding is None else padding

    assert not (padding is None and border), "must have padding > 0 when requesting border"

    n = board.N
    sample = np.zeros([2, n, n], dtype=np.uint8)

    current = board.current_color
    if switch:
        current = 1 - current
    for move in board.stones:
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
        size = board.N
        a_border = (padding - 1) * [0] + (size + 2) * [1] + (padding - 1) * [0]
        other_layer[padding - 1] = a_border
        other_layer[size + padding] = a_border
        other_layer[:, padding - 1] = a_border
        other_layer[:, size + padding] = a_border

    both = np.array([current_layer, other_layer])

    return np.rollaxis(both, 0, 3).astype(float)


def create_sample(stones, n, viewpoint):
    
    sample = np.zeros([2, n, n], dtype=np.uint8)

    current = WHITE
    for move in stones:
        r, c = gtools.b2m(move, n)
        sample[current][r][c] = 1
        current = 1 - current

    offensive = np.hstack([
        np.zeros([n + 2, 1], dtype=np.uint8),
        np.vstack([np.zeros(n, dtype=np.uint8),
                   sample[viewpoint],
                   np.zeros(n, dtype=np.uint8)]),
        np.zeros([n + 2, 1], dtype=np.uint8)
    ])
    
    defensive = np.hstack([
        np.ones([n + 2, 1], dtype=np.uint8),
        np.vstack([np.ones(n, dtype=np.uint8),
                   sample[1-viewpoint],
                   np.ones(n, dtype=np.uint8)]),
        np.ones([n + 2, 1], dtype=np.uint8)
    ])
    both = np.array([offensive, defensive])
    return np.rollaxis(both, 0, 3)


def wrap_sample(array, value):
    n = np.shape(array)[0]
    return np.hstack([
        np.zeros([n+2, 1], dtype=np.float32) + value,
        np.vstack([np.zeros(n, dtype=np.float32) + value,
                   array, 
                   np.zeros(n, dtype=np.float32) + value]),
        np.zeros([n+2, 1], dtype=np.float32) + value
    ])


def create_samples_and_qvalues(board, policy, heuristics):
    """
    create 8 equivalent samples and qvalues from the given board
    """
    from copy import deepcopy
    
    # A little tweak: If the last move saw an "immediate win", 
    # we won't need the board that has a finished line of 5.
    if True:
        dcp = deepcopy(board)
        dcp.undo()
        if policy.suggest(dcp).status == 2:
            board.undo().undo()

    all_stones_t = [transform(board.stones.copy(), board.N, rot, ref)
                    for rot in range(4)
                    for ref in [False, True]]

    samples = []
    qvalues = []
    avalues = []
    for stones_t in all_stones_t:
        sample = create_sample(stones_t, board.N, 1-board.current_color)
        board = GomokuBoard(heuristics, board.N, stones=stones_t)
        qvalue, default_value = heuristic_QF(board, policy)
        qvalue = wrap_sample(qvalue, default_value)
        samples.append(sample)
        qvalues.append(qvalue)
        avalues.append((qvalue-default_value)/100.0)

    return (np.array(samples), 
            np.reshape(qvalues, [8, board.N+2, board.N+2, 1]), 
            np.reshape(avalues, [8, board.N+2, board.N+2, 1]))


def data_from_game(board, policy, heuristics):    
    """
    Careful: This function rolls back the board
    """
    # Don't want to see fours (my heuristics don't work well when the game is essentially done anyway.)
    board.undo(False).undo(False)

    s, q, a = create_samples_and_qvalues(board, policy, heuristics)
    while board.cursor > 6:
        board.undo()
        s1, q1, a1 = create_samples_and_qvalues(board, policy, heuristics)
        s = np.concatenate((s, s1))
        q = np.concatenate((q, q1))
        a = np.concatenate((a, a1))
    return s, q, a


def to_matrix12(sample):
    field = np.rollaxis(sample.reshape(22, 22, 2), 2, 0).astype(np.int)
    unwrapped = (field[0]+field[1]*2)[1:-1].T[1:-1].T
    return unwrapped


def to_matrix_xo(sample):
    if np.sum(to_matrix12(sample) > 0) % 2 == 0:
        symbols = ['. ', 'x ', 'o ']
    else:
        symbols = ['. ', 'o ', 'x ']
    im12 = to_matrix12(sample)
    return "\n".join(["".join([symbols[c] for c in im12[r]]) for r in range(20)])

#########################################################################################################
#
#               NEW STUFF BELOW!!
#
#########################################################################################################


def create_nxnx4(size: int, stones=None, pad_r=0, pad_l=0, pad_t=0, pad_b=0,
                 padding=None, border=False, switch=False):
    """
    Creates a NxNx4 NDArray from the stones of the board. Black is in the 0-plane
    """
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
    sample[row][column][0] = 1
    n = sample.shape[0]
    zeros = np.zeros((n, n))
    new_board = np.rollaxis(np.stack([sample[:, :, 1], sample[:, :, 0], zeros, zeros]), 0, 3)
    return new_board
