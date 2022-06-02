import numpy as np

import tensorflow as tf
from domoku.policies.radial import all_2xnxn

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


class MaxCriticalityPolicy(tf.keras.Model):
    """
    A policy that doesn't miss any sure-win or must-defend
    """
    def __init__(self, input_size, overconfidence=1., **kwargs):
        """
        :param input_size: n of the board
        :param overconfidence: Any value above 1 prefers offensive play to to defender's benefit.
        :param kwargs:
        """

        self.input_size = input_size
        super().__init__(**kwargs)

        self.patterns = [
            # terminal_pattern
            [
                [[0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], -4, [1000, 500]],
            ],
            # win-in-1 patterns
            [
                [[0, 1, 1, 1, 1, -1, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0], -3, [99, 50]],
                [[0, 0, 1, 1, 1, -1, 1, 0, 0, 0, 0], [0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0], -3, [99, 50]],
                [[0, 0, 0, 1, 1, -1, 1, 1, 0, 0, 0], [0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0], -3, [99, 50]],
                [[0, 0, 0, 0, 1, -1, 1, 1, 1, 0, 0], [0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0], -3, [99, 50]],
                [[0, 0, 0, 0, 0, -1, 1, 1, 1, 1, 0], [0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0], -3, [99, 50]],
            ],
            # win-in-2 patterns
            [
                [[0, -1, 1, 1, 1, -1, -1, 0, 0, 0, 0], [0, -1, 0, 0, 0, -1, -1, 0, 0, 0, 0], -2, [10, 5]],
                [[0, 0, -1, 1, 1, -1, 1, -1, 0, 0, 0], [0, 0, -1, 0, 0, -1, 0, -1, 0, 0, 0], -2, [10, 5]],
                [[0, 0, 0, -1, 1, -1, 1, 1, -1, 0, 0], [0, 0, 0, -1, 0, -1, 0, 0, -1, 0, 0], -2, [10, 5]],
                [[0, 0, 0, 0, -1, -1, 1, 1, 1, -1, 0], [0, 0, 0, 0, -1, -1, 0, 0, 0, -1, 0], -2, [10, 5]],
            ],

            # potential double-open 3 patterns - not so critical
            # Here, the over-confidence may help us to produce terminating trajectories
            [
                [[+0, -1,  1, -1,  1, -1, -1,  0,  0,  0,  0], [+0, -1, -1, -1, -1, -1, -1,  0,  0,  0,  0],
                 -1, [overconfidence, 1]],
                [[+0,  0, -1,  1, -1, -1,  1, -1,  0,  0,  0], [+0,  0, -1, -1, -1, -1, -1, -1,  0,  0,  0],
                 -1, [overconfidence, 1]],
                [[+0,  0,  0, -1,  1, -1, -1,  1, -1,  0,  0], [+0,  0,  0, -1, -1, -1, -1, -1, -1,  0,  0],
                 -1, [overconfidence, 1]],
                [[+0,  0,  0,  0, -1, -1,  1, -1,  1, -1,  0], [+0,  0,  0,  0, -1, -1, -1, -1, -1, -1,  0],
                 -1, [overconfidence, 1]],

                [[+0, -1,  1,  1, -1, -1, -1,  0,  0,  0,  0], [+0, -1, -1, -1, -1, -1, -1,  0,  0,  0,  0],
                 -1, [overconfidence, 1]],
                [[+0,  0, -1,  1,  1, -1, -1, -1,  0,  0,  0], [+0,  0, -1, -1, -1, -1, -1, -1,  0,  0,  0],
                 -1, [overconfidence, 1]],
                [[+0,  0,  0, -1,  1, -1,  1, -1, -1,  0,  0], [+0,  0,  0, -1, -1, -1, -1, -1, -1,  0,  0],
                 -1, [overconfidence, 1]],
                [[+0,  0,  0,  0, -1, -1,  1,  1, -1, -1,  0], [+0,  0,  0,  0, -1, -1, -1, -1, -1, -1,  0],
                 -1, [overconfidence, 1]],
            ],

            # potential single-open-4 patterns - They are important in threat sequences
            [
                #  x o . o o []
                [[+0,  1, -1,  1,  1,    -1,    0,  0,  0,  0,  0], [1, -1, -1, -1, -1, -1, 0,  0,  0,  0,  0],
                 -3, [overconfidence, 1]],
                #  x o o . o []
                [[+0,  1,  1, -1,  1,    -1,    0,  0,  0,  0,  0], [1, -1, -1, -1, -1, -1, 0,  0,  0,  0,  0],
                 -3, [overconfidence, 1]],
                #  x o o o . []
                [[+0,  1,  1,  1,  -1,    -1,   0,  0,  0,  0,  0], [1, -1, -1, -1, -1, -1, 0,  0,  0,  0,  0],
                 -3, [overconfidence, 1]],

                # - x o . o [] o .
                [[0,  0,  1, -1,  1,     -1,    1,  0,  0,  0,  0], [0, +1, -1, -1, -1, -1, -1, 0,  0,  0,  0],
                 -3, [overconfidence, 1]],
                #  - x o o . [] o
                [[0,  0,  1,  1,  -1,    -1,    1,  0,  0,  0,  0], [0, +1, -1, -1, -1, -1, -1, 0,  0,  0,  0],
                 -3, [overconfidence, 1]],
                #  - x . o o [] o
                [[0,  0,  -1, 1,  1,     -1,    1,  0,  0,  0,  0], [0, +1, -1, -1, -1, -1, -1, 0,  0,  0,  0],
                 -3, [overconfidence, 1]],

                # [] o o . o x
                [[0,  0,  0,  0,  0,    -1,   1, 1, -1, 1, 0], [0,  0,  0,  0,  0, -1, -1, -1, -1, -1, 1],
                 -3, [overconfidence, 1]],
                # [] o . o o x
                [[0,  0,  0,  0,  0,    -1,   1, -1, 1, 1, 0], [0,  0,  0,  0,  0, -1, -1, -1, -1, -1, 1],
                 -3, [overconfidence, 1]],
                # [] . o o o x
                [[0,  0,  0,  0,  0,    -1,   -1, 1, 1, 1, 0], [0,  0,  0,  0,  0, -1, -1, -1, -1, -1, 1],
                 -3, [overconfidence, 1]],

                # o [] o . o x -
                [[0,  0,  0,  0,  1,     -1,    1, -1,  1, 0, 0], [0,  0,  0,  0,  0, -1, -1, -1, -1, 1, 0],
                 -3, [overconfidence, 1]],
                # o [] o o . x -
                [[0,  0,  0,  0,  1,     -1,    1,  1, -1, 0, 0], [0,  0,  0,  0,  0, -1, -1, -1, -1, 1, 0],
                 -3, [overconfidence, 1]],
                # o [] . o o x -
                [[0,  0,  0,  0,  1,     -1,    -1, 1,  1, 0, 0], [0,  0,  0,  0,  0, -1, -1, -1, -1, 1, 0],
                 -3, [overconfidence, 1]],
            ]

        ]

        filters, biases, weights = self.assemble_filters()

        self.detector = tf.keras.layers.Conv2D(
            filters=len(biases), kernel_size=(11, 11),
            kernel_initializer=tf.constant_initializer(filters),
            bias_initializer=tf.constant_initializer(biases),
            activation=tf.nn.relu,
            padding='same',
            input_shape=(input_size, input_size, 2))

        self.combine = tf.keras.layers.Conv2D(
            filters=1, kernel_size=(1, 1),
            kernel_initializer=tf.constant_initializer(weights))

    def call(self, state):
        # Allow other representations here by ignoring their additional channels
        if state.shape[-1] == 4:
            state = state[:, :, :2]

        state = np.reshape(state, [-1, self.input_size, self.input_size, 2])
        res = self.combine(self.detector(state))
        res = tf.clip_by_value(res, 0, 1000)
        return res


    def winner(self, sample):
        max_crit = np.max(self.call(sample), axis=None)
        return 0 if max_crit > 900 else 1 if max_crit > 400 else None

        #
    #  All about constructing the convolutional filters down from here
    #


    def select_patterns(self, channel: int = None, criticality: int = None):
        channels = [channel] if channel is not None else CHANNELS
        criticalities = [criticality] if criticality is not None else CRITICALITIES

        patterns = [
            [[offense, defense], bias, weights[0]] if channel == CURRENT else [[defense, offense], bias, weights[1]]
            for criticality in criticalities
            for offense, defense, bias, weights in self.patterns[criticality]
            for channel in channels
        ]
        patterns.sort(key=lambda r: -r[-1])
        return patterns


    def assemble_filters(self):
        patterns = self.select_patterns()
        biases = []
        weights = []
        for pattern in patterns:
            biases = biases + [pattern[1]] * 4
            weights = weights + [pattern[2]] * 4
        stacked = np.stack([
            all_2xnxn(pattern[0])
            for pattern in patterns], axis=3)
        reshaped = np.reshape(stacked, (11, 11, 2, 4 * np.shape(patterns)[0]))

        return reshaped, biases, weights
