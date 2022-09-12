from pickle import Unpickler
from typing import List, Tuple, Any, Callable

import numpy as np
import tensorflow as tf

from aegomoku.gomoku_board import GomokuBoard
from aegomoku.gomoku_game import GomokuGame


def read_training_data(filename: str, condition: Callable = None):
    """
    Reads training data as
        stones: array(np.uint8) of flat board positions of subsequent moves up until that state
        probabilities: array of all probabilities on a uint8 scale 0-255
        values: float
    from file and returns eight (N+2)x(N+2) position symmetries and the corresponding probabilities and values
    as a straight array of (state, probabiltiy (float), value(float)) triples
    :param condition: a filter condition (s, p, v)->bool
    :param filename: file name
    :return: Array of (state, probabiltiy (float), value(float)) triples and an list of the game trajectories
    """
    all_examples = []
    all_games = []
    with open(filename, "rb") as file:
        while True:
            try:
                trajectory = Unpickler(file).load()
                all_examples += expand_trajectory(trajectory, condition)
                first, stones, _ = trajectory
                all_games.append((first, stones))

            except EOFError:
                return all_examples, all_games


def expand_trajectory(trajectory, condition: Callable = None):
    """
    read the training data from the trajectory record, ignore name and stones. They're for analysis purposes.
    :param trajectory: A record like (name, stones, positions) where positions are like (stones, probs, value)
    :param condition: a predicate function that defines whether to consider a particular tuple of (stones, prob, value)
    :return: fully expanded (from symmetries) examples (state, pred, value)
    """
    examples = []

    # We keep supporting trajectories without headers
    try:
        _, _, trajectory = trajectory
    except ValueError as e:
        trajectory = trajectory

    for position in trajectory:
        stones, probs, value = position
        if condition is not None:
            if not condition(stones, probs, value):
                continue
        board_size = np.sqrt(probs.shape[0]).astype(int)
        game = GomokuGame(board_size)
        probabilities = probs / 255.
        state = GomokuBoard(board_size, stones).canonical_representation()
        symmetries = game.get_symmetries(state, probabilities)
        for state, prediction in symmetries:
            examples.append((state, prediction, value))
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
