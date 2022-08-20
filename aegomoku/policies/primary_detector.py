import logging
from typing import Callable

import numpy as np
import tensorflow as tf

from aegomoku.policies.radial import all_3xnxn, radial_3xnxn

logger = logging.getLogger(__name__)

# Criticality Categories
TERMINAL = 0  # detects existing 5-rows
WIN_IN_1 = 1  # detects positions that create/prohibit rows of 5
WIN_IN_2 = 2  # detects positions that create/prohibit double-open 4-rows
DO_3 = 3  # detects positions that create/prohibit a 3-row with two open ends
SO_4 = 4  # detects positions that create/prohibit a 4-row with a single open end

CRITICALITIES = [
    TERMINAL, WIN_IN_1, WIN_IN_2, DO_3, SO_4
]

CURRENT = 0
OTHER = 1
CHANNELS = [
    CURRENT, OTHER
]


class PrimaryDetector(tf.keras.layers.Layer, Callable):
    """
    Input:  Board plus boundary: dimensions: [N+2, N+2, 3]
    Output: Projections + current threat plus other threat: [N+2, N+2, 5]
    """
    def __init__(self, board_size, activation=None, **kwargs):
        """
        :param board_size: length of the board including the boundary
        :param kwargs:
        """
        self.input_size = board_size + 2  # We include the boundary in the input
        super().__init__(**kwargs)

        # weights for offensive vs defensive choices
        w_o = 2.
        w_d = 1.

        self.patterns = [
            # terminal_pattern
            [
                [[0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], -4, [9999, 3333]],
            ],
            # win-in-1 patterns
            [
                [[0, 1, 1, 1, 1, -1, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0], -3, [999, 333]],
                [[0, 0, 1, 1, 1, -1, 1, 0, 0, 0, 0], [0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0], -3, [999, 333]],
                [[0, 0, 0, 1, 1, -1, 1, 1, 0, 0, 0], [0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0], -3, [999, 333]],
                [[0, 0, 0, 0, 1, -1, 1, 1, 1, 0, 0], [0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0], -3, [999, 333]],
                [[0, 0, 0, 0, 0, -1, 1, 1, 1, 1, 0], [0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0], -3, [999, 333]],
            ],
            # win-in-2 patterns
            [
                # maybe not so weak as defense patterns
                [[-1, 1, -1, 1, 1, -1,  0, 0, 0, 0, 0], [-1, 0, -1, 0, 0, -1,  0, 0, 0, 0, 0], -2, [9, 27]],
                [[-1, 1, 1, -1, 1, -1,  0, 0, 0, 0, 0], [-1, 0, 0, -1, 0, -1,  0, 0, 0, 0, 0], -2, [9, 27]],

                [[0, -1, 1, 1, 1, -1, -1, 0, 0, 0, 0], [0, -1, 0, 0, 0, -1, -1, 0, 0, 0, 0], -2, [99, 33]],
                [[0, 0, -1, 1, 1, -1, 1, -1, 0, 0, 0], [0, 0, -1, 0, 0, -1, 0, -1, 0, 0, 0], -2, [99, 33]],
                [[0, 0, 0, -1, 1, -1, 1, 1, -1, 0, 0], [0, 0, 0, -1, 0, -1, 0, 0, -1, 0, 0], -2, [99, 33]],
                [[0, 0, 0, 0, -1, -1, 1, 1, 1, -1, 0], [0, 0, 0, 0, -1, -1, 0, 0, 0, -1, 0], -2, [99, 33]],

                # maybe not so weak as defense patterns
                [[0, 0, 0, 0, -1, -1, 1, 1, -1, 1, -1], [0, 0, 0, 0, 0, -1, 0, 0, -1, 0, -1], -2, [9, 27]],
                [[0, 0, 0, 0, -1, -1, 1, -1, 1, 1, -1], [0, 0, 0, 0, 0, -1, 0, -1, 0, 0, -1], -2, [9, 27]],
            ],

            # potential double-open 3 patterns - not so critical
            [
                [[+0, -1,  1, -1,  1, -1, -1,  0,  0,  0,  0], [+0, -1, -1, -1, -1, -1, -1,  0,  0,  0,  0],
                 -1, [w_o, w_d]],
                [[+0,  0, -1,  1, -1, -1,  1, -1,  0,  0,  0], [+0,  0, -1, -1, -1, -1, -1, -1,  0,  0,  0],
                 -1, [w_o, w_d]],
                [[+0,  0,  0, -1,  1, -1, -1,  1, -1,  0,  0], [+0,  0,  0, -1, -1, -1, -1, -1, -1,  0,  0],
                 -1, [w_o, w_d]],
                [[+0,  0,  0,  0, -1, -1,  1, -1,  1, -1,  0], [+0,  0,  0,  0, -1, -1, -1, -1, -1, -1,  0],
                 -1, [w_o, w_d]],

                [[+0, -1,  1,  1, -1, -1, -1,  0,  0,  0,  0], [+0, -1, -1, -1, -1, -1, -1,  0,  0,  0,  0],
                 -1, [w_o, w_d]],
                [[+0,  0, -1,  1,  1, -1, -1, -1,  0,  0,  0], [+0,  0, -1, -1, -1, -1, -1, -1,  0,  0,  0],
                 -1, [w_o, w_d]],
                [[+0,  0,  0, -1,  1, -1,  1, -1, -1,  0,  0], [+0,  0,  0, -1, -1, -1, -1, -1, -1,  0,  0],
                 -1, [w_o, w_d]],
                [[+0,  0,  -1, -1,  1, -1,  1, -1, 0,  0,  0], [+0,  0,  -1, -1, -1, -1, -1, -1, 0,  0,  0],
                 -1, [w_o, w_d]],
                [[+0,  0,  0,  0, -1, -1,  1,  1, -1, -1,  0], [+0,  0,  0,  0, -1, -1, -1, -1, -1, -1,  0],
                 -1, [w_o, w_d]],
                [[+0,  0,  0,  0, -1, -1,  -1, 1,  1, -1,  0], [+0,  0,  0,  0, -1, -1, -1, -1, -1, -1,  0],
                 -1, [w_o, w_d]],
            ],

            # potential single-open-4 patterns - They are important in threat sequences
            [
                #  x o . o o []
                [[+0,  1, -1,  1,  1,    -1,    0,  0,  0,  0,  0], [1, -1, -1, -1, -1, -1, 0,  0,  0,  0,  0],
                 -3, [w_o, w_d]],
                #  x o o . o []
                [[+0,  1,  1, -1,  1,    -1,    0,  0,  0,  0,  0], [1, -1, -1, -1, -1, -1, 0,  0,  0,  0,  0],
                 -3, [w_o, w_d]],
                #  x o o o . []
                [[+0,  1,  1,  1,  -1,    -1,   0,  0,  0,  0,  0], [1, -1, -1, -1, -1, -1, 0,  0,  0,  0,  0],
                 -3, [w_o, w_d]],

                #  - x o o o [] .
                [[0,  0,  1,  1,  1,     -1,   -1,  0,  0,  0,  0], [0, +1, -1, -1, -1, -1, -1, 0,  0,  0,  0],
                 -3, [w_o, w_d]],
                #  - x o . o [] o
                [[0,  0,  1, -1,  1,     -1,    1,  0,  0,  0,  0], [0, +1, -1, -1, -1, -1, -1, 0,  0,  0,  0],
                 -3, [w_o, w_d]],
                #  - x o o . [] o
                [[0,  0,  1,  1,  -1,    -1,    1,  0,  0,  0,  0], [0, +1, -1, -1, -1, -1, -1, 0,  0,  0,  0],
                 -3, [w_o, w_d]],
                #  - x . o o [] o
                [[0,  0,  -1, 1,  1,     -1,    1,  0,  0,  0,  0], [0, +1, -1, -1, -1, -1, -1, 0,  0,  0,  0],
                 -3, [w_o, w_d]],

                #  [] o o . o x
                [[0,  0,  0,  0,  0,    -1,   1, 1, -1, 1, 0], [0,  0,  0,  0,  0, -1, -1, -1, -1, -1, 1],
                 -3, [w_o, w_d]],
                #  [] o . o o x
                [[0,  0,  0,  0,  0,    -1,   1, -1, 1, 1, 0], [0,  0,  0,  0,  0, -1, -1, -1, -1, -1, 1],
                 -3, [w_o, w_d]],
                #  [] . o o o x
                [[0,  0,  0,  0,  0,    -1,   -1, 1, 1, 1, 0], [0,  0,  0,  0,  0, -1, -1, -1, -1, -1, 1],
                 -3, [w_o, w_d]],

                #  o [] o . o x -
                [[0,  0,  0,  0,  1,     -1,    1, -1,  1, 0, 0], [0,  0,  0,  0,  0, -1, -1, -1, -1, 1, 0],
                 -3, [w_o, w_d]],
                #  o [] o o . x -
                [[0,  0,  0,  0,  1,     -1,    1,  1, -1, 0, 0], [0,  0,  0,  0,  0, -1, -1, -1, -1, 1, 0],
                 -3, [w_o, w_d]],
                #  o [] . o o x -
                [[0,  0,  0,  0,  1,     -1,    -1, 1,  1, 0, 0], [0,  0,  0,  0,  0, -1, -1, -1, -1, 1, 0],
                 -3, [w_o, w_d]],
                #  . [] o o o x -
                [[0,  0,  0,  0, -1,     -1,     1, 1,  1, 0, 0], [0,  0,  0,  0,  -1, -1, -1, -1, -1, 1, 0],
                 -3, [w_o, w_d]],
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

        # Layer 2. Output: curr / other / boundary / inf_curr / inf_other
        weights = self.spread_weights(weights, n_filters)
        weights = np.rollaxis(np.array(weights), axis=-1)

        self.combine = tf.keras.layers.Conv2D(
            name='heuristic_priority',
            filters=5, kernel_size=(1, 1),
            activation=activation,
            trainable=False,
            kernel_initializer=tf.constant_initializer(weights))

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
        res2 = self.combine(res1)
        return res2

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
        biases = []
        weights = []

        projectors = self.get_projectors()

        for pattern in patterns:
            biases = biases + [pattern[1]] * 4
            weights = weights + [pattern[2]] * 4

        stacked = np.stack([
            all_3xnxn(pattern[0])
            for pattern in patterns], axis=3)
        reshaped = np.reshape(stacked, (11, 11, 3, 4 * np.shape(patterns)[0]))

        reshaped = np.concatenate([projectors, reshaped], axis=3)

        return reshaped, [0, 0, 0] + biases, [1, 1, 1] + weights


    @staticmethod
    def get_projectors(len_radial: int = 5):
        """
        Projectors simply pass the stone channels through to the next layer
        :return: Three projector filters for the three input channels
        """
        proj_cur = radial_3xnxn([0] * len_radial, None, None, 1, 0, 0)
        proj_oth = radial_3xnxn([0] * len_radial, None, None, 0, 1, 0)
        proj_bnd = radial_3xnxn([0] * len_radial, None, None, 0, 0, 1)
        filters = [proj_cur, proj_oth, proj_bnd]
        return np.stack(filters, axis=3)
