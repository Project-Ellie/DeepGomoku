from typing import Callable

import numpy as np

import tensorflow as tf
from domoku.constants import BLACK, WHITE


def new_sample(board_size: int, num_blacks: int = 0, num_whites: int = 0):
    """
    Creates a board rep
    :param board_size: The n of the board
    :param num_blacks:
    :param num_whites:
    :return:
    """
    sample = np.zeros([board_size, board_size, 2])
    for n in range(num_blacks):
        row, col = np.random.randint(0, board_size, 2)
        sample[row, col, BLACK] = 1
    for n in range(num_whites):
        row, col = np.random.randint(0, board_size, 2)
        sample[row, col, WHITE] = 1
    return sample


def new_dataset(size: int, sampler: Callable, labeler: tf.keras.Model, separate=False):

    samples = []
    labels = []
    for i in range(size):
        a_sample = sampler()
        samples.append(a_sample)
        label = np.squeeze(labeler(np.expand_dims(a_sample, axis=0)))
        labels.append(label)

    if separate:
        return np.array(samples), np.array(labels)
    else:
        return tf.data.Dataset.from_tensor_slices((samples, labels))
