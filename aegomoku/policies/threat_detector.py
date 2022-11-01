import logging
from typing import Callable

import numpy as np
import tensorflow as tf

from aegomoku.policies.radial import all_3xnxn, radial_3xnxn

logger = logging.getLogger(__name__)

# Criticality Categories: Nonmenclature reflects what the position could result in.

# The terminators: MUST ACT
IMMEDIATE_WIN = 0       # detects LO4 positions that create/prohibit LO5
DOUBLE_OPEN_4 = 1       # detects LO3 positions that create/prohibit double-open LO4

# Elements of threat sequences: Opponent MUST REACT
# Consider equal here - MCTS will learn to distinguish
SINGLE_OPEN_4 = 2       # detects LO3 positions that allow for single-open LO4
THREATENING_DO4 = 3     # detects LO2 positions that can stage a DO4 in the next step

# Getting started
VICINITY = 4            # detects positions in the vicinity of other stones

CRITICALITIES = [
    IMMEDIATE_WIN, DOUBLE_OPEN_4, SINGLE_OPEN_4, THREATENING_DO4, VICINITY
]

CURRENT = 0
OTHER = 1
CHANNELS = [
    CURRENT, OTHER
]


class ThreatDetector(tf.keras.layers.Layer, Callable):
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

        self.patterns = [
            # IMMEDIATE_WIN
            [
                [[1, 1, 1, 1, -1, 0, 0, 0, 0],
                 [0, 0, 0, 0, -1, 0, 0, 0, 0], -3],

                [[0, 1, 1, 1, -1, 1, 0, 0, 0],
                 [0, 0, 0, 0, -1, 0, 0, 0, 0], -3],

                [[0, 0, 1, 1, -1, 1, 1, 0, 0],
                 [0, 0, 0, 0, -1, 0, 0, 0, 0], -3],

                [[0, 0, 0, 1, -1, 1, 1, 1, 0],
                 [0, 0, 0, 0, -1, 0, 0, 0, 0], -3],

                [[0, 0, 0, 0, -1, 1, 1, 1, 1],
                 [0, 0, 0, 0, -1, 0, 0, 0, 0], -3],
            ],

            # DOUBLE_OPEN_4
            [
                # . . . - _ x x x -
                [[0, 0, 0, -1, -1, 1, 1, 1, -1],
                 [0, 0, 0, -1, -1, 0, 0, 0, -1], -2],

                # . . - x _ x x - .
                [[0, 0, -1, 1, -1, 1, 1, -1, 0],
                 [0, 0, -1, 0, -1, 0, 0, -1, 0], -2],

                # . - x x _ x - . .
                [[0, -1,  1,  1, -1, 1, -1,  0, 0],
                 [0, -1,  0,  0, -1, 0, -1,  0, 0], -2],

                # - x x x _ - . . .
                [[-1,  1,  1,  1, -1, -1,  0,  0, 0],
                 [-1,  0,  0,  0, -1, -1,  0,  0, 0], -2]
            ],

            # SINGLE_OPEN_4
            [
                # With defensive stones
                # o x x x _ - . . .
                [[0,  1,  1,  1,  -1, -1,  0,  0,  0],
                 [1,  0,  0,  0,  -1, -1,  0,  0,  0], -3],
                # . . . - _ x x x o
                [[0,  0,  0, -1, -1,  1,  1,  1,  0],
                 [0,  0,  0, -1, -1,  0,  0,  0,  1], -3],

                # . o x x _ x - . .
                [[0,  0,  1,  1, -1,  1, -1,  0,  0],
                 [0,  1,  0,  0, -1,  0, -1,  0,  0], -3],
                # . . - x _ x x o .
                [[0,  0, -1,  1, -1,  1,  1,  0,  0],
                 [0,  0, -1,  0, -1,  0,  0,  1,  0], -3],

                # . . o x _ x x - .
                [[0,  0,  0,  1, -1,  1,  1, -1,  0],
                 [0,  0,  1,  0, -1,  0,  0, -1,  0], -3],
                # . - x x _ x o . .
                [[0, -1,  1,  1, -1,  1,  0,  0,  0],
                 [0, -1,  0,  0, -1,  0,  1,  0,  0], -3],

                # . . . o _ x x x -
                [[0,  0,  0,  0, -1,  1,  1,  1, -1],
                 [0,  0,  0,  1, -1,  0,  0,  0, -1], -3],
                # - x x x _ o . . .
                [[-1,  1,  1,  1, -1,  0,  0,  0,  0],
                 [-1,  0,  0,  0, -1,  1,  0,  0,  0], -3],


                # spread across five fields
                # x x x - _ . . . .
                [[1,  1,  1, -1, -1,  0,  0,  0,  0],
                 [0,  0,  0, -1, -1,  0,  0,  0,  0], -2],
                # . . . . _ - x x x
                [[0,  0,  0,  0, -1, -1,  1,  1,  1],
                 [0,  0,  0,  0, -1, -1,  0,  0,  0], -2],

                # x x - x _ . . . .
                [[1,  1, -1,  1, -1,  0,  0,  0,  0],
                 [0,  0, -1,  0, -1,  0,  0,  0,  0], -2],
                # . . . . _ x - x x
                [[0,  0,  0,  0, -1,  1, -1,  1,  1],
                 [0,  0,  0,  0, -1,  0, -1,  0,  0], -2],

                # x - x x _ . . . .
                [[1, -1,  1,  1, -1,  0,  0,  0,  0],
                 [0, -1,  0,  0, -1,  0,  0,  0,  0], -2],
                # . . . . _ x x - x
                [[0,  0,  0,  0, -1,  1,  1, -1,  1],
                 [0,  0,  0,  0, -1,  0,  0, -1,  0], -2],

                # . x x - _ x . . .
                [[0,  1,  1, -1, -1,  1,  0,  0,  0],
                 [0,  0,  0, -1, -1,  0,  0,  0,  0], -2],
                # . . . x _ - x x .
                [[0,  0,  0,  1, -1, -1,  1,  1,  0],
                 [0,  0,  0,  0, -1, -1,  0,  0,  0], -2],

                # . x - x _ x . . .
                [[0,  1, -1,  1, -1,  1,  0,  0,  0],
                 [0,  0, -1,  0, -1,  0,  0,  0,  0], -2],
                # . . . x _ x - x .
                [[0,  0,  0,  1, -1,  1, -1,  1,  0],
                 [0,  0,  0,  0, -1,  0, -1,  0,  0], -2],

                # . . x x _ - x . .
                [[0,  0,  1,  1, -1, -1,  1,  0,  0],
                 [0,  0,  0,  0, -1, -1,  0,  0,  0], -2],
                # . . x - _ x x . .
                [[0,  0,  1, -1, -1,  1,  1,  0,  0],
                 [0,  0,  0, -1, -1,  0,  0,  0,  0], -2],

            ],
            # THREATENING_DO4
            [
                # - x x - _ - . . .
                [[-1,  1,  1, -1, -1, -1,  0,  0,  0],
                 [-1,  0,  0, -1, -1, -1,  0,  0,  0], -1],
                # . . . - _ - x x -
                [[0,  0,  0, -1, -1, -1,  1,  1, -1],
                 [0,  0,  0, -1, -1, -1,  0,  0, -1], -1],

                # - x - x _ - . . .
                [[-1,  1, -1,  1, -1,  -1,  0,  0,  0],
                 [-1,  0, -1,  0, -1,  -1,  0,  0,  0], -1],
                # . . . - _ x - x -
                [[0,  0,  0, -1, -1,  1, -1,  1, -1],
                 [0,  0,  0, -1, -1,  0, -1,  0, -1], -1],

                # - - x x _ - . . .
                [[-1, -1,  1,  1, -1, -1,  0,  0,  0],
                 [-1, -1,  0,  0, -1, -1,  0,  0,  0], -1],
                # . . . - _ x x - -
                [[0,  0,  0, -1, -1,  1,  1, -1, -1],
                 [0,  0,  0, -1, -1,  0,  0, -1, -1], -1],

                # . - x x _ - - . .
                [[0, -1,  1,  1, -1,  -1, -1,  0,  0],
                 [0, -1,  0,  0, -1,  -1, -1,  0,  0], -1],
                # . . - - _ x x - .
                [[0,  0, -1, -1, -1,  1,  1, -1, -1],
                 [0,  0, -1, -1, -1,  0,  0, -1,  0], -1],

                # . - - x _ x - . .
                [[0, -1, -1,  1, -1,  1, -1,  0,  0],
                 [0, -1, -1,  0, -1,  0, -1,  0,  0], -1],
                # . . - x _ x - - .
                [[0,  0, -1,  1, -1,  1, -1, -1,  0],
                 [0,  0, -1,  0, -1,  0, -1, -1,  0], -1],

                # . - x - _ x - . .
                [[0, -1,  1, -1, -1,  1, -1,  0,  0],
                 [0, -1,  0, -1, -1,  0, -1,  0,  0], -1],
                # . . - x _ - x - .
                [[0,  0, -1,  1, -1, -1,  1, -1,  0],
                 [0,  0, -1,  0, -1, -1,  0, -1,  0], -1],

            ],
            # VICINITY
            [
                # Anything in the vicinity? No logic - just anything next or single-hop
                [[0, -1,  1, -1, -1, -1,  0,  0,  0],
                 [0,  0,  0,  0, -1,  0,  0,  0,  0], 0],

                [[0, -1, -1,  1, -1, -1, -1,  0,  0],
                 [0,  0,  0,  0, -1,  0,  0,  0,  0], 0],

                [[0,  0, -1, -1, -1,  1, -1, -1,  0],
                 [0,  0,  0,  0, -1,  0,  0,  0,  0], 0],

                [[0,  0,  0, -1, -1, -1,  1, -1,  0],
                 [0,  0,  0,  0, -1,  0,  0,  0,  0], 0],
            ]
        ]

        filters, biases = self.assemble_filters()

        n_filters = len(biases)

        # Layer 1. Output: (curr/oth) x 4 directions x num patterns
        self.detector = tf.keras.layers.Conv2D(
            name="heuristic_detector",
            filters=n_filters, kernel_size=(9, 9),
            kernel_initializer=tf.constant_initializer(filters),
            bias_initializer=tf.constant_initializer(biases),
            activation=tf.nn.relu,
            padding='same',
            trainable=False)

        # Layer 2. Output: curr / other / boundary / inf_curr / inf_other
        weights = self.spread_weights()
        weights = np.rollaxis(np.array(weights), axis=-1)

        self.combine = tf.keras.layers.Conv2D(
            name='heuristic_priority',
            filters=10, kernel_size=(1, 1),
            activation=activation,
            trainable=False,
            kernel_initializer=tf.constant_initializer(weights))

    @staticmethod
    def spread_weights():
        """
        distribute the weights
        5 threat levels x 2 colors
        """
        i0 = [1, 1, 1, 1, 0, 0, 0, 0] * 5 + [0] * 320
        j0 = [0, 0, 0, 0, 1, 1, 1, 1] * 5 + [0] * 320

        i1 = [0] * 40 + [1, 1, 1, 1, 0, 0, 0, 0] * 4 + [0] * 288
        j1 = [0] * 40 + [0, 0, 0, 0, 1, 1, 1, 1] * 4 + [0] * 288

        i2 = [0] * 72 + [1, 1, 1, 1, 0, 0, 0, 0] * 20 + [0] * 128
        j2 = [0] * 72 + [0, 0, 0, 0, 1, 1, 1, 1] * 20 + [0] * 128

        i3 = [0] * 232 + [1, 1, 1, 1, 0, 0, 0, 0] * 12 + [0] * 32
        j3 = [0] * 232 + [0, 0, 0, 0, 1, 1, 1, 1] * 12 + [0] * 32

        i4 = [0] * 328 + [1, 1, 1, 1, 0, 0, 0, 0] * 4
        j4 = [0] * 328 + [0, 0, 0, 0, 1, 1, 1, 1] * 4

        return [i0, j0, i1, j1, i2, j2, i3, j3, i4, j4]

    def call(self, state):
        """
        Each channel of the return value represents one of 10 threat levels. The value of a field is 1 if at least
        one line brings about that threat level.
        :param state: state, representated as (n+2) x (n+2) x 3 board with boundary
        :return: (n+2) x (n+2) x 10 feature matrix:

        """
        # States are nxnx3
        # try:
        #     state = np.reshape(state, [-1, self.input_size, self.input_size, 3]).astype(float)
        # except ValueError as e:
        #     logger.error(f"Got {type(state)}: {state}. Why?")
        #     raise e

        res1 = self.detector(state)
        res2 = self.combine(res1)
        return tf.sign(res2)

    #
    #  All about constructing the convolutional filters down from here
    #


    def select_patterns(self, channel: int = None, criticality: int = None):
        channels = [channel] if channel is not None else CHANNELS
        criticalities = [criticality] if criticality is not None else CRITICALITIES

        patterns = [
            [[offense, defense, defense], bias]
            if channel == CURRENT
            else [[defense, offense, defense], bias]
            for criticality in criticalities
            for offense, defense, bias in self.patterns[criticality]
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

        for pattern in patterns:
            biases = biases + [pattern[1]] * 4

        stacked = np.stack([
            all_3xnxn(pattern[0])
            for pattern in patterns], axis=3)
        reshaped = np.reshape(stacked, (9, 9, 3, 4 * np.shape(patterns)[0]))

        return reshaped, biases
