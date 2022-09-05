import logging
from typing import Callable

import numpy as np
import tensorflow as tf

from aegomoku.policies.radial import all_3xnxn

logger = logging.getLogger(__name__)

# Criticality Categories
TERMINAL = 0  # detects existing 5-rows
WIN_IN_1 = 1  # detects positions that create/prohibit rows of 5

CRITICALITIES = [
    TERMINAL, WIN_IN_1
]

CURRENT = 0
OTHER = 1
CHANNELS = [
    CURRENT, OTHER
]


class TerminalDetector(tf.keras.layers.Layer, Callable):
    """
    Input:  Board plus boundary: dimensions: [N+2, N+2, 3]
    Output: Projections + current threat plus other threat: [N+2, N+2, 5]
    """
    def __init__(self, board_size, **kwargs):
        """
        :param board_size: length of the board including the boundary
        :param kwargs:
        """
        self.input_size = board_size + 2  # We include the boundary in the input
        super().__init__(**kwargs)

        self.patterns = [  # TODO:
            # terminal_pattern
            [
                [[0, 0, -1, 1, 1, 1, 1, 1, -1, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], -4],
            ],
            # win-in-1 patterns
            [
                [[-5, 1, 1, 1, 1, -5, -5, 0, 0, 0, 0], [0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0], -3],
                [[0, -5, 1, 1, 1, -5, 1, -5, 0, 0, 0], [0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0], -3],
                [[0, 0, -5, 1, 1, -5, 1, 1, -5, 0, 0], [0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0], -3],
                [[0, 0, 0, -5, 1, -5, 1, 1, 1, -5, 0], [0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0], -3],
                [[0, 0, 0, 0, -5, -5, 1, 1, 1, 1, -5], [0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0], -3],
            ]
        ]

        filters, biases, weights = self.assemble_filters()

        n_filters = len(biases)

        # Layer 1. Output: (curr/oth) x 4 directions x 32 patterns + 3 projectors => 259 channels per board
        self.detector = tf.keras.layers.Conv2D(
            name="heuristic_detector",
            filters=n_filters, kernel_size=(11, 11),
            kernel_initializer=tf.constant_initializer(filters),
            bias_initializer=tf.constant_initializer(biases),
            activation=tf.nn.relu,
            padding='same',
            trainable=False)


    @staticmethod
    def spread_weights(weights, n_channels):
        """
        distribute the weights across 5 different combiner 1x1xn_channels filters
        """
        n_patterns = (n_channels - 3) // 8

        # current stones projected
        c = [1] + [0] * (n_channels - 1)

        # other stones projected
        o = [0, 1] + [0] * (n_channels - 2)

        # boundary stones projected
        b = [0, 0, 1] + [0] * (n_channels - 3)

        # total influence of the current stones
        i = [0, 0, 0] + list(np.array([1, 1, 1, 1, 0, 0, 0, 0] * n_patterns) * weights[3:])

        # total influence of the other stones
        j = [0, 0, 0] + list(np.array([0, 0, 0, 0, 1, 1, 1, 1] * n_patterns) * weights[3:])

        return [c, o, b, i, j]

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
        return res1

    #
    #  All about constructing the convolutional filters down from here
    #


    def select_patterns(self, channel: int = None, criticality: int = None):
        channels = [channel] if channel is not None else CHANNELS
        criticalities = [criticality] if criticality is not None else CRITICALITIES

        patterns = [
            [[offense, defense, defense], bias, weights[0]]
            if channel == CURRENT
            else [[defense, offense, defense], bias, weights[1]]
            for criticality in criticalities
            for offense, defense, bias, weights in self.patterns[criticality]
            for channel in channels
        ]

        return patterns


    def assemble_filters(self):
        """
        Considering the boundary stones just as good a defense as one of the opponent's stone.
        Boundary stones are placed on the periphery of the 3rd channel
        """
        patterns = self.select_patterns()

        biases = [-4] * 8 + [-3] * 40
        weights = [1] * 40

        stacked = np.stack([
            all_3xnxn(pattern[0])
            for pattern in patterns], axis=3)
        reshaped = np.reshape(stacked, (11, 11, 3, 4 * np.shape(patterns)[0]))

        return reshaped, biases, weights
