import logging
from typing import Callable

import numpy as np
import tensorflow as tf

from aegomoku.policies.radial import all_3xnxn, radial_3xnxn

logger = logging.getLogger(__name__)

# Criticality Categories
IMMEDIATE_THREAT = 0    # detects lines of 4 that create/prohibit rows of 5
THREAT_OPPORTUNITY = 1  # detects lines of 3 that lead to immediate threats
THREAT_POTENTIAL = 2    # detects lines of 2 that lead to threat opportunities
VICINITY = 3            # detects any stones in the vicinity
CRITICALITIES = [
    IMMEDIATE_THREAT, THREAT_OPPORTUNITY, THREAT_POTENTIAL, VICINITY
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

            # threat ooportunities
            [
                [[1,  1,  1, -1, -1,  0,  0,  0,  0],
                 [0,  0,  0, -1, -1,  0,  0,  0,  0], -2],

                [[1,  1, -1,  1, -1,  0,  0,  0,  0],
                 [0,  0, -1,  0, -1,  0,  0,  0,  0], -2],

                [[1, -1,  1,  1, -1,  0,  0,  0,  0],
                 [0, -1,  0,  0, -1,  0,  0,  0,  0], -2],

                [[-1,  1,  1,  1, -1,  0,  0,  0,  0],
                 [-1,  0,  0,  0, -1,  0,  0,  0,  0], -2],


                [[0,  1,  1,  1, -1, -1,  0,  0,  0],
                 [0,  0,  0,  0, -1, -1,  0,  0,  0], -2],

                [[0,  1,  1, -1, -1,  1,  0,  0,  0],
                 [0,  0,  0, -1, -1,  0,  0,  0,  0], -2],

                [[0,  1, -1,  1, -1,  1,  0,  0,  0],
                 [0,  0, -1,  0, -1,  0,  0,  0,  0], -2],

                [[0, -1,  1,  1, -1,  1,  0,  0,  0],
                 [0, -1,  0,  0, -1,  0,  0,  0,  0], -2],


                [[0,  0,  1,  1, -1,  1, -1,  0,  0],
                 [0,  0,  0,  0, -1,  0, -1,  0,  0], -2],

                [[0,  0,  1,  1, -1, -1,  1,  0,  0],
                 [0,  0,  0,  0, -1, -1,  0,  0,  0], -2],

                [[0,  0,  1, -1, -1,  1,  1,  0,  0],
                 [0,  0,  0, -1, -1,  0,  0,  0,  0], -2],

                [[0,  0, -1,  1, -1,  1,  1,  0,  0],
                 [0,  0, -1,  0, -1,  0,  0,  0,  0], -2],


                [[0,  0,  0,  1, -1,  1,  1, -1,  0],
                 [0,  0,  0,  0, -1,  0,  0, -1,  0], -2],

                [[0,  0,  0,  1, -1,  1, -1,  1,  0],
                 [0,  0,  0,  0, -1,  0, -1,  0,  0], -2],

                [[0,  0,  0,  1, -1, -1,  1,  1,  0],
                 [0,  0,  0,  0, -1, -1,  0,  0,  0], -2],

                [[0,  0,  0, -1, -1,  1,  1,  1,  0],
                 [0,  0,  0, -1, -1,  0,  0,  0,  0], -2],


                [[0,  0,  0,  0, -1,  1,  1,  1, -1],
                 [0,  0,  0,  0, -1,  0,  0,  0, -1], -2],

                [[0,  0,  0,  0, -1,  1,  1, -1,  1],
                 [0,  0,  0,  0, -1,  0,  0, -1,  0], -2],

                [[0,  0,  0,  0, -1,  1, -1,  1,  1],
                 [0,  0,  0,  0, -1,  0, -1,  0,  0], -2],

                [[0,  0,  0,  0, -1, -1,  1,  1,  1],
                 [0,  0,  0,  0, -1, -1,  0,  0,  0], -2]

            ], [

                [[-1,  1,  1, -1, -1, -1,  0,  0,  0],
                 [-1,  0,  0, -1, -1, -1,  0,  0,  0], -1],

                [[-1,  1, -1,  1, -1,  -1,  0,  0,  0],
                 [-1,  0, -1,  0, -1,  -1,  0,  0,  0], -1],

                [[-1, -1,  1,  1, -1,  -1, -1,  0,  0],
                 [0,  -1,  0,  0, -1,  -1, -1,  0,  0], -1],

                [[-1, -1,  1,  1, -1, -1,  0,  0,  0],
                 [-1, -1,  0,  0, -1, -1,  0,  0,  0], -1],


                [[0, -1,  1, -1, -1,  1, -1, -1,  0],
                 [0, -1,  0, -1, -1,  0, -1,  0,  0], -1],

                [[0, -1, -1,  1, -1,  1, -1,  0,  0],
                 [0, -1, -1,  0, -1,  0, -1,  0,  0], -1],

                [[0,  0, -1,  1, -1,  1, -1, -1,  0],
                 [0,  0, -1,  0, -1,  0, -1, -1,  0], -1],

                [[0,  0, -1,  1, -1, -1,  1, -1,  0],
                 [0,  0, -1,  0, -1, -1,  0, -1,  0], -1],


                # . . . - _ x x - -
                [[0,  0,  0, -1, -1,  1,  1, -1, -1],
                 [0,  0,  0, -1, -1,  0,  0, -1, -1], -1],

                # . . - - _ x x - .
                [[0,  0, -1, -1, -1,  1,  1, -1, -1],
                 [0,  0, -1, -1, -1,  0,  0, -1,  0], -1],

                # . . . - _ x - x -
                [[0,  0,  0, -1, -1,  1, -1,  1, -1],
                 [0,  0,  0, -1, -1,  0, -1,  0, -1], -1],

                # . . . - _ - x x -
                [[0,  0,  0, -1, -1, -1,  1,  1, -1],
                 [0,  0,  0, -1, -1, -1,  0,  0, -1], -1]
            ], [

                # Anything in the vicinity? No logic - just anything
                [[-1, -1,  1, -1, -1, -1, -1, -1, -1],
                 [0,   0,  0,  0, -1,  0,  0,  0,  0], 0],

                [[-1, -1, -1,  1, -1, -1, -1, -1, -1],
                 [0,   0,  0,  0, -1,  0,  0,  0,  0], 0],

                [[-1, -1, -1, -1, -1,  1, -1, -1, -1],
                 [0,   0,  0,  0, -1,  0,  0,  0,  0], 0],

                [[-1, -1, -1, -1, -1, -1,  1, -1, -1],
                 [0,   0,  0,  0, -1,  0,  0,  0,  0], 0],
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
            filters=8, kernel_size=(1, 1),
            activation=activation,
            trainable=False,
            kernel_initializer=tf.constant_initializer(weights))

    @staticmethod
    def spread_weights():
        """
        distribute the weights
        """
        # threat of the current player
        i0 = [1, 1, 1, 1, 0, 0, 0, 0] * 5 + [0] * 288

        # threat of the other player
        j0 = [0, 0, 0, 0, 1, 1, 1, 1] * 5 + [0] * 288

        # opportunities of the current player
        i1 = [0] * 40 + [1, 1, 1, 1, 0, 0, 0, 0] * 20 + [0] * 128

        # opportunities of the other player
        j1 = [0] * 40 + [0, 0, 0, 0, 1, 1, 1, 1] * 20 + [0] * 128

        # potential positions of current player
        i2 = [0] * 200 + [1, 1, 1, 1, 0, 0, 0, 0] * 12 + [0] * 32

        # opportunities of the other stones
        j2 = [0] * 200 + [0, 0, 0, 0, 1, 1, 1, 1] * 12 + [0] * 32

        i3 = [0] * 296 + [1, 1, 1, 1, 0, 0, 0, 0] * 4

        j3 = [0] * 296 + [0, 0, 0, 0, 1, 1, 1, 1] * 4

        return [i0, j0, i1, j1, i2, j2, i3, j3]

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
