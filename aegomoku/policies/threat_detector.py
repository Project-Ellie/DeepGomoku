import logging
from typing import Callable

import numpy as np
import tensorflow as tf

from aegomoku.policies.radial import all_3xnxn, radial_3xnxn

logger = logging.getLogger(__name__)

# Criticality Categories
WIN_IN_1 = 0  # detects positions that create/prohibit rows of 5
WIN_IN_2 = 1  # detects positions that create/prohibit double-open 4-rows

CRITICALITIES = [
    WIN_IN_1, WIN_IN_2
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
            # immediate threats
            [
                [[1, 1, 1, 1, -1, 0, 0, 0, 0],
                 [0, 0, 0, 0, -1, 0, 0, 0, 0], -3, [999, 333]],

                [[0, 1, 1, 1, -1, 1, 0, 0, 0],
                 [0, 0, 0, 0, -1, 0, 0, 0, 0], -3, [999, 333]],

                [[0, 0, 1, 1, -1, 1, 1, 0, 0],
                 [0, 0, 0, 0, -1, 0, 0, 0, 0], -3, [999, 333]],

                [[0, 0, 0, 1, -1, 1, 1, 1, 0],
                 [0, 0, 0, 0, -1, 0, 0, 0, 0], -3, [999, 333]],

                [[0, 0, 0, 0, -1, 1, 1, 1, 1],
                 [0, 0, 0, 0, -1, 0, 0, 0, 0], -3, [999, 333]],
            ],

            # threat ooportunities
            [
                [[1,  1,  1, -1, -1,  0,  0,  0,  0],
                 [0,  0,  0, -1, -1,  0,  0,  0,  0], -2, [99, 33]],

                [[1,  1, -1,  1, -1,  0,  0,  0,  0],
                 [0,  0, -1,  0, -1,  0,  0,  0,  0], -2, [99, 33]],

                [[1, -1,  1,  1, -1,  0,  0,  0,  0],
                 [0, -1,  0,  0, -1,  0,  0,  0,  0], -2, [99, 33]],

                [[-1,  1,  1,  1, -1,  0,  0,  0,  0],
                 [-1,  0,  0,  0, -1,  0,  0,  0,  0], -2, [99, 33]],


                [[0,  1,  1,  1, -1, -1,  0,  0,  0],
                 [0,  0,  0,  0, -1,  0,  0,  0,  0], -2, [99, 33]],

                [[0,  1,  1, -1, -1,  1,  0,  0,  0],
                 [0,  0,  0,  0, -1,  0,  0,  0,  0], -2, [99, 33]],

                [[0,  1, -1,  1, -1,  1,  0,  0,  0],
                 [0,  0,  0,  0, -1,  0,  0,  0,  0], -2, [99, 33]],

                [[0, -1,  1,  1, -1,  1,  0,  0,  0],
                 [0,  0,  0,  0, -1,  0,  0,  0,  0], -2, [99, 33]],


                [[0,  0,  1,  1, -1,  1, -1,  0,  0],
                 [0,  0,  0,  0, -1,  0, -1,  0,  0], -2, [99, 33]],

                [[0,  0,  1,  1, -1, -1,  1,  0,  0],
                 [0,  0,  0,  0, -1, -1,  0,  0,  0], -2, [99, 33]],

                [[0,  0,  1, -1, -1,  1,  1,  0,  0],
                 [0,  0,  0, -1, -1,  1,  0,  0,  0], -2, [99, 33]],

                [[0,  0, -1,  1, -1,  1,  1,  0,  0],
                 [0,  0, -1,  0, -1,  0,  0,  0,  0], -2, [99, 33]],


                [[0,  0,  0,  1, -1,  1,  1, -1,  0],
                 [0,  0,  0,  0, -1,  0,  0, -1,  0], -2, [99, 33]],

                [[0,  0,  0,  1, -1,  1, -1,  1,  0],
                 [0,  0,  0,  0, -1,  0, -1,  0,  0], -2, [99, 33]],

                [[0,  0,  0,  1, -1, -1,  1,  1,  0],
                 [0,  0,  0,  0, -1, -1,  0,  0,  0], -2, [99, 33]],

                [[0,  0,  0, -1, -1,  1,  1,  1,  0],
                 [0,  0,  0, -1, -1,  0,  0,  0,  0], -2, [99, 33]],


                [[0,  0,  0,  0, -1,  1,  1,  1, -1],
                 [0,  0,  0,  0, -1,  0,  0,  0, -1], -2, [99, 33]],

                [[0,  0,  0,  0, -1,  1,  1, -1,  1],
                 [0,  0,  0,  0, -1,  0,  0, -1,  0], -2, [99, 33]],

                [[0,  0,  0,  0, -1,  1, -1,  1,  1],
                 [0,  0,  0,  0, -1,  0, -1,  0,  0], -2, [99, 33]],

                [[0,  0,  0,  0, -1, -1,  1,  1,  1],
                 [0,  0,  0,  0, -1, -1,  0,  0,  0], -2, [99, 33]],
            ]

        ]

        filters, biases, weights = self.assemble_filters()

        n_filters = len(biases)

        # Layer 1. Output: (curr/oth) x 4 directions x 32 patterns + 3 projectors => 259 channels per board
        self.detector = tf.keras.layers.Conv2D(
            name="heuristic_detector",
            filters=n_filters, kernel_size=(9, 9),
            kernel_initializer=tf.constant_initializer(filters),
            bias_initializer=tf.constant_initializer(biases),
            activation=tf.nn.relu,
            padding='same',
            trainable=False)

        # Layer 2. Output: curr / other / boundary / inf_curr / inf_other
        weights = self.spread_weights(weights)
        weights = np.rollaxis(np.array(weights), axis=-1)

        self.combine = tf.keras.layers.Conv2D(
            name='heuristic_priority',
            filters=4, kernel_size=(1, 1),
            activation=activation,
            trainable=False,
            kernel_initializer=tf.constant_initializer(weights))

    @staticmethod
    def spread_weights(weights):
        """
        distribute the weights
        """
        # threat of the current stones
        i0 = list(np.array([1, 1, 1, 1, 0, 0, 0, 0] * 5) * weights[:40]) + [0] * 160

        # threat of the other stones
        j0 = list(np.array([0, 0, 0, 0, 1, 1, 1, 1] * 5) * weights[:40]) + [0] * 160

        # opportunities of the current stones
        i1 = [0] * 40 + list(np.array([1, 1, 1, 1, 0, 0, 0, 0] * 20) * weights[40:])

        # opportunities of the other stones
        j1 = [0] * 40 + list(np.array([0, 0, 0, 0, 1, 1, 1, 1] * 20) * weights[40:])

        return [i0, j0, i1, j1]

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

        for pattern in patterns:
            biases = biases + [pattern[1]] * 4
            weights = weights + [pattern[2]] * 4

        stacked = np.stack([
            all_3xnxn(pattern[0])
            for pattern in patterns], axis=3)
        reshaped = np.reshape(stacked, (9, 9, 3, 4 * np.shape(patterns)[0]))

        return reshaped, biases, weights


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
