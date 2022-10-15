import logging

import numpy as np
import tensorflow as tf

from aegomoku.interfaces import TerminalDetector
from aegomoku.policies.radial import all_3xnxn

logger = logging.getLogger(__name__)


class GomokuTerminalDetector(tf.keras.layers.Layer, TerminalDetector):
    """
    Note that a line of 6 does NOT win!
    Input:  Board plus boundary: dimensions: [N+2, N+2, 3]
    Output: the number of rows of 5 on current and other layer
        example [0, 1]: other player won. This is the result after the terminal move
    """
    def get_winner(self, state):
        res = self.call(state)
        if res[1] > 0:
            return 1
        elif res[0] > 0:
            return 0
        else:
            return None

    def __init__(self, board_size, allow_overlines=False, **kwargs):
        """
        :param board_size: length of the board including the boundary
        :param kwargs:
        """
        self.input_size = board_size + 2  # We include the boundary in the input
        super().__init__(**kwargs)

        protector = 0 if allow_overlines else -1

        patterns = [[[protector, 1, 1, 1, 1, 1, protector],
                     [0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0]],
                    [[0, 0, 0, 0, 0, 0, 0],
                     [protector, 1, 1, 1, 1, 1, protector],
                     [0, 0, 0, 0, 0, 0, 0]]]

        stacked = np.stack([
            all_3xnxn(pattern)
            for pattern in patterns], axis=3)
        reshaped = np.reshape(stacked, (7, 7, 3, 4 * np.shape(patterns)[0]))

        # Layer 1. Output: (curr/oth) x 4 directions x 32 patterns + 3 projectors => 259 channels per board
        self.detector = tf.keras.layers.Conv2D(
            name="Terminal_detector",
            filters=8,
            kernel_size=(7, 7),
            kernel_initializer=tf.constant_initializer(reshaped),
            bias_initializer=tf.constant_initializer(-4.),
            activation=tf.nn.relu,
            padding='same',
            trainable=False)

        curr_other_filter = [[1, 0], [1, 0], [1, 0], [1, 0], [0, 1], [0, 1], [0, 1], [0, 1]]

        self.collector = tf.keras.layers.Conv2D(
            name="Collector",
            filters=2,
            kernel_size=(1, 1),
            kernel_initializer=tf.constant_initializer(curr_other_filter),
            bias_initializer=tf.constant_initializer(0.),
            trainable=False)

    def call(self, state):
        """
        :param state: state, representated as (n+2) x (n+2) x 3 board with boundary
        :return: the logit, clipped
        """
        # States are nxnx3
        # try:
        #     state = np.reshape(state, [-1, self.input_size, self.input_size, 3]).astype(float)
        # except ValueError as e:
        #     logger.error(f"Got {type(state)}: {state}. Why?")
        #     raise e

        res1 = self.detector(state)
        res1 = self.collector(res1)
        return tf.reduce_sum(res1, axis=[0, 1, 2])
