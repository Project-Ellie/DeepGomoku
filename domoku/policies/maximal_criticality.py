import numpy as np
import tensorflow as tf

from alphazero.interfaces import TerminalDetector
from domoku.policies.radial import all_2xnxn, all_3xnxn

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


class MaxCriticalityPolicy(tf.keras.Model, TerminalDetector):
    """
    A policy that doesn't miss any sure-win or must-defend
    """
    def __init__(self, board_size, overconfidence=1., **kwargs):
        """
        :param board_size: length of the board including the boundary
        :param overconfidence: Any value above 1 prefers offensive play to to defender's benefit.
        :param kwargs:
        """

        self.input_size = board_size + 2  # We include the boundary in the input
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

        filters, biases, weights = self.assemble_filters_3()

        self.detector = tf.keras.layers.Conv2D(
            filters=len(biases), kernel_size=(11, 11),
            kernel_initializer=tf.constant_initializer(filters),
            bias_initializer=tf.constant_initializer(biases),
            activation=tf.nn.relu,
            padding='same',
            input_shape=(board_size, board_size, 3))

        self.combine = tf.keras.layers.Conv2D(
            filters=1, kernel_size=(1, 1),
            kernel_initializer=tf.constant_initializer(weights))

    def call(self, state):
        """
        :param state: state, representated as (n+2) x (n+2) x 3 board with boundary
        :return: the logit, clipped
        """
        # States are nxnx3
        state = np.reshape(state, [-1, self.input_size, self.input_size, 3]).astype(float)
        res = self.combine(self.detector(state))
        res = tf.clip_by_value(res, 0, 1000)
        return res

    def q_p_v(self, state):
        q = tf.nn.tanh(self.call(state))
        p = tf.nn.softmax(q)
        v = tf.reduce_max(q)
        return q, p, v


    def get_winner(self, sample):
        max_crit = np.max(self.call(sample), axis=None)
        return 0 if max_crit > 900 else 1 if max_crit > 400 else None


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


    def assemble_filters_3(self):
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
        reshaped = np.reshape(stacked, (11, 11, 3, 4 * np.shape(patterns)[0]))

        return reshaped, biases, weights
