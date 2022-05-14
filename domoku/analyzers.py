from dataclasses import dataclass
from typing import List

import numpy as np
import tensorflow as tf
from domoku.board import GomokuBoard
from domoku.data import create_binary_rep


DOWNDIAG = np.eye(5, dtype=float)
UPDIAG = DOWNDIAG[::-1]

NONE = np.zeros([5, 5], dtype=float)
HOR = NONE.copy()
HOR[2] = 1
VER = NONE.copy()
VER[:, 2] = 1


@dataclass
class Container:
    UP: List[np.array]
    DOWN: List[np.array]
    HOR: List[np.array]
    VER: List[np.array]


@dataclass
class FilterContainer:
    BLACK: Container
    WHITE: Container


PATTERN = FilterContainer(
    BLACK=Container(UP=[UPDIAG, NONE], DOWN=[DOWNDIAG, NONE],
                    HOR=[HOR, NONE], VER=[VER, NONE]),
    WHITE=Container(UP=[NONE, UPDIAG], DOWN=[NONE, DOWNDIAG],
                    HOR=[NONE, HOR], VER=[NONE, VER]))


class Analyzer:

    def __init__(self, n):
        self.n = n

        filters = np.array([
            PATTERN.WHITE.UP,
            PATTERN.WHITE.DOWN,
            PATTERN.WHITE.HOR,
            PATTERN.WHITE.VER,
            PATTERN.BLACK.UP,
            PATTERN.BLACK.DOWN,
            PATTERN.BLACK.HOR,
            PATTERN.BLACK.VER
        ])
        self.filters = np.rollaxis(np.rollaxis(filters, 1, 4), 0, 4)

        kernel_init = tf.constant_initializer(self.filters)
        bias_init = tf.constant_initializer(-4.)

        self.model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(filters=len(filters), kernel_size=(5, 5),
                                   kernel_initializer=kernel_init, bias_initializer=bias_init,
                                   activation=tf.nn.relu,
                                   input_shape=(self.n + 4, self.n + 4, 2))])


    def detect_five(self, board: GomokuBoard):
        """
        Detects whether the given board features a line of 5 of the given color
        :param board: the board to be analyzed
        """
        # create a padding to simplify detection near the borders
        sample = create_binary_rep(board, pad_r=2, pad_t=2, pad_b=2, pad_l=2)
        sample = np.reshape(sample, [-1, self.n + 4, self.n + 4, 2])
        recognized = self.model(sample)
        detected = np.squeeze(recognized.numpy())
        return detected == 1
