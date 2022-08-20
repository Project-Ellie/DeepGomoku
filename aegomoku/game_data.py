from pickle import Unpickler
from typing import List, Tuple, Any

import numpy as np
import tensorflow as tf

from aegomoku.gomoku_board import GomokuBoard
from aegomoku.gomoku_game import GomokuGame


def read_training_data(filename: str, board_size: int):
    """
    Reads training data as
        stones: array(np.uint8) of flat board positions of subsequent moves up until that state
        probabilities: array of all probabilities on a uint8 scale 0-255
        values: float
    from file and returns eight (N+2)x(N+2) position symmetries and the corresponding probabilities and values
    as a straight array of (state, probabiltiy (float), value(float)) triples
    :param filename: file name
    :param board_size: board size
    :return: Array of (state, probabiltiy (float), value(float)) triples
    """
    examples = []
    game = GomokuGame(board_size, None)
    with open(filename, "rb") as file:
        while True:
            try:
                traj = Unpickler(file).load()
                for position in traj:
                    stones, probs, value = position
                    probabilities = probs / 255.
                    state = GomokuBoard(board_size, stones).canonical_representation()
                    symmetries = game.get_symmetries(state, probabilities)
                    for state, prediction in symmetries:
                        examples.append((state, prediction, value))
            except EOFError:
                return examples


def create_dataset(data: List[Tuple[Any, Any, Any]], batch_size=1024, shuffle=True):
    subset = data
    x = np.asarray([t[0] for t in subset], dtype=float)
    pi = np.asarray([t[1] for t in subset])
    v = np.asarray([t[2] for t in subset])
    x_ds = tf.data.Dataset.from_tensor_slices(x).batch(batch_size)
    pi_ds = tf.data.Dataset.from_tensor_slices(pi).batch(batch_size)
    v_ds = tf.data.Dataset.from_tensor_slices(v).batch(batch_size)
    all_ds = tf.data.Dataset.zip((x_ds, pi_ds, v_ds))
    if shuffle:
        all_ds = all_ds.shuffle(buffer_size=batch_size)
    return all_ds
