from typing import Any, Callable

import numpy as np

import tensorflow as tf
from domoku.constants import BLACK, WHITE


def new_sample(board_size: int, num_blacks: int = 0, num_whites: int = 0):
    """
    Creates a board rep
    :param board_size: The size of the board
    :param num_blacks:
    :param num_whites:
    :return:
    """
    sample = np.zeros([1, board_size, board_size, 2])
    for n in range(num_blacks):
        row, col = np.random.randint(0, board_size, 2)
        sample[0, row, col, BLACK] = 1
    for n in range(num_whites):
        row, col = np.random.randint(0, board_size, 2)
        sample[0, row, col, WHITE] = 1
    return sample


def new_dataset(size: int, sampler: Callable, labeler: tf.keras.Model):

    samples = []
    labels = []
    for i in range(size):
        a_sample = sampler()
        samples.append(a_sample)
        labels.append(np.squeeze(labeler(a_sample)))
    return tf.data.Dataset.from_tensor_slices((samples, labels))
