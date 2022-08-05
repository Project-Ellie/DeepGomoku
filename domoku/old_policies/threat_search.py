import numpy as np
import tensorflow as tf

from aegomoku.interfaces import TerminalDetector
from domoku.policies.radial import all_3xnxn, radial_3xnxn, all_5xnxn

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


class ThreatSearchPolicy(tf.keras.Model, TerminalDetector):
    """
    A policy that doesn't miss any sure-win or must-defend
    """
    def __init__(self, board_size, **kwargs):
        """
        :param board_size: length of the board including the boundary
        :param kwargs:
        """

        self.input_size = board_size + 2  # We include the boundary in the input
        super().__init__(**kwargs)

        self.patterns = [
            # terminal_pattern
            [
                [[0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], -4, [9991, 1333]],
            ],
            # win-in-1 patterns
            [
                [[0, 1, 1, 1, 1, -1, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0], -3, [991, 133]],
                [[0, 0, 1, 1, 1, -1, 1, 0, 0, 0, 0], [0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0], -3, [999, 333]],
                [[0, 0, 0, 1, 1, -1, 1, 1, 0, 0, 0], [0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0], -3, [999, 333]],
                [[0, 0, 0, 0, 1, -1, 1, 1, 1, 0, 0], [0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0], -3, [999, 333]],
                [[0, 0, 0, 0, 0, -1, 1, 1, 1, 1, 0], [0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0], -3, [999, 333]],
            ],
            # win-in-2 patterns
            [
                [[0, -1, 1, 1, 1, -1, -1, 0, 0, 0, 0], [0, -1, 0, 0, 0, -1, -1, 0, 0, 0, 0], -2, [99, 33]],
                [[0, 0, -1, 1, 1, -1, 1, -1, 0, 0, 0], [0, 0, -1, 0, 0, -1, 0, -1, 0, 0, 0], -2, [99, 33]],
                [[0, 0, 0, -1, 1, -1, 1, 1, -1, 0, 0], [0, 0, 0, -1, 0, -1, 0, 0, -1, 0, 0], -2, [99, 33]],
                [[0, 0, 0, 0, -1, -1, 1, 1, 1, -1, 0], [0, 0, 0, 0, -1, -1, 0, 0, 0, -1, 0], -2, [99, 33]],
            ],

            # potential double-open 3 patterns - not so critical
            [
                [[+0, -1,  1, -1,  1, -1, -1,  0,  0,  0,  0], [+0, -1, -1, -1, -1, -1, -1,  0,  0,  0,  0],
                 -1, [1, 1]],
                [[+0,  0, -1,  1, -1, -1,  1, -1,  0,  0,  0], [+0,  0, -1, -1, -1, -1, -1, -1,  0,  0,  0],
                 -1, [1, 1]],
                [[+0,  0,  0, -1,  1, -1, -1,  1, -1,  0,  0], [+0,  0,  0, -1, -1, -1, -1, -1, -1,  0,  0],
                 -1, [1, 1]],
                [[+0,  0,  0,  0, -1, -1,  1, -1,  1, -1,  0], [+0,  0,  0,  0, -1, -1, -1, -1, -1, -1,  0],
                 -1, [1, 1]],

                [[+0, -1,  1,  1, -1, -1, -1,  0,  0,  0,  0], [+0, -1, -1, -1, -1, -1, -1,  0,  0,  0,  0],
                 -1, [1, 1]],
                [[+0,  0, -1,  1,  1, -1, -1, -1,  0,  0,  0], [+0,  0, -1, -1, -1, -1, -1, -1,  0,  0,  0],
                 -1, [1, 1]],
                [[+0,  0,  0, -1,  1, -1,  1, -1, -1,  0,  0], [+0,  0,  0, -1, -1, -1, -1, -1, -1,  0,  0],
                 -1, [1, 1]],
                [[+0,  0,  -1, -1,  1, -1,  1, -1, 0,  0,  0], [+0,  0,  -1, -1, -1, -1, -1, -1, 0,  0,  0],
                 -1, [1, 1]],
                [[+0,  0,  0,  0, -1, -1,  1,  1, -1, -1,  0], [+0,  0,  0,  0, -1, -1, -1, -1, -1, -1,  0],
                 -1, [1, 1]],
                [[+0,  0,  0,  0, -1, -1,  -1, 1,  1, -1,  0], [+0,  0,  0,  0, -1, -1, -1, -1, -1, -1,  0],
                 -1, [1, 1]],
            ],

            # potential single-open-4 patterns - They are important in threat sequences
            [
                #  x o . o o []
                [[+0,  1, -1,  1,  1,    -1,    0,  0,  0,  0,  0], [1, -1, -1, -1, -1, -1, 0,  0,  0,  0,  0],
                 -3, [1, 1]],
                #  x o o . o []
                [[+0,  1,  1, -1,  1,    -1,    0,  0,  0,  0,  0], [1, -1, -1, -1, -1, -1, 0,  0,  0,  0,  0],
                 -3, [1, 1]],
                #  x o o o . []
                [[+0,  1,  1,  1,  -1,    -1,   0,  0,  0,  0,  0], [1, -1, -1, -1, -1, -1, 0,  0,  0,  0,  0],
                 -3, [1, 1]],

                # - x o . o [] o .
                [[0,  0,  1, -1,  1,     -1,    1,  0,  0,  0,  0], [0, +1, -1, -1, -1, -1, -1, 0,  0,  0,  0],
                 -3, [1, 1]],
                #  - x o o . [] o
                [[0,  0,  1,  1,  -1,    -1,    1,  0,  0,  0,  0], [0, +1, -1, -1, -1, -1, -1, 0,  0,  0,  0],
                 -3, [1, 1]],
                #  - x . o o [] o
                [[0,  0,  -1, 1,  1,     -1,    1,  0,  0,  0,  0], [0, +1, -1, -1, -1, -1, -1, 0,  0,  0,  0],
                 -3, [1, 1]],

                # [] o o . o x
                [[0,  0,  0,  0,  0,    -1,   1, 1, -1, 1, 0], [0,  0,  0,  0,  0, -1, -1, -1, -1, -1, 1],
                 -3, [1, 1]],
                # [] o . o o x
                [[0,  0,  0,  0,  0,    -1,   1, -1, 1, 1, 0], [0,  0,  0,  0,  0, -1, -1, -1, -1, -1, 1],
                 -3, [1, 1]],
                # [] . o o o x
                [[0,  0,  0,  0,  0,    -1,   -1, 1, 1, 1, 0], [0,  0,  0,  0,  0, -1, -1, -1, -1, -1, 1],
                 -3, [1, 1]],

                # o [] o . o x -
                [[0,  0,  0,  0,  1,     -1,    1, -1,  1, 0, 0], [0,  0,  0,  0,  0, -1, -1, -1, -1, 1, 0],
                 -3, [1, 1]],
                # o [] o o . x -
                [[0,  0,  0,  0,  1,     -1,    1,  1, -1, 0, 0], [0,  0,  0,  0,  0, -1, -1, -1, -1, 1, 0],
                 -3, [1, 1]],
                # o [] . o o x -
                [[0,  0,  0,  0,  1,     -1,    -1, 1,  1, 0, 0], [0,  0,  0,  0,  0, -1, -1, -1, -1, 1, 0],
                 -3, [1, 1]],
            ]
        ]

        # Patterns to recognize combinations of potential stones with existing stones
        self.secondary_patterns = [

            # - X x - [] - . . .
            [[-1,  1, -1, -1, -99,   -1,  0,  0,  0],
             [0,   0,  1,  0,   1,    0,  0,  0,  0],
             [-1, -1, -1, -1,  -1,   -1,  0,  0,  0]],

            # - X - x [] - . . .
            [[-1,  1, -1, -1, -99,   -1,  0,  0,  0],
             [0,   0,  0,  1,   1,    0,  0,  0,  0],
             [-1, -1, -1, -1,  -1,   -1,  0,  0,  0]],

            # - x X - [] - . . .
            [[-1, -1, 1, -1,  -99,   -1,  0,  0,  0],
             [0,   1,  0, 0,    1,    0,  0,  0,  0],
             [-1, -1, -1, -1,  -1,   -1,  0,  0,  0]],

            # - - X x [] - - . .
            [[-1, -1,  1, -1, -99,   -1, -1,  0,  0],
             [0,   0,  0,  1,   1,    0,  0,  0,  0],
             [-1, -1, -1, -1,  -1,   -1, -1,  0,  0]],

            # . - X - [] x - . .
            [[0, -1,  1, -1, -99,   -1, -1,  0,  0],
             [0,  0,  0,  0,   1,    1,  0,  0,  0],
             [0, -1, -1, -1,  -1,   -1, -1,  0,  0]],

            # - x - X [] - . . .
            [[-1, -1, -1, 1,  -99,   -1,  0,  0,  0],
             [0,   1,  0,  0,   1,   0,  0,  0,  0],
             [-1, -1, -1, -1,  -1,   -1,  0,  0,  0]],

            # - - x X [] - - . .
            [[-1, -1, -1, 1,  -99,   -1, -1,  0,  0],
             [0,   0, 1,  0,    1,    0,  0,  0,  0],
             [-1, -1, -1, -1,  -1,   -1, -1,  0,  0]],

            # . - - X [] x - - .
            [[0, -1, -1,  1, -99,   -1, -1, -1,  0],
             [0,  0,  0,  0,   1,    1,  0,  0,  0],
             [0, -1, -1, -1,  -1,   -1, -1, -1,  0]],

            # . . - X [] - x - .
            [[0,  0, -1,  1, -99,   -1, -1, -1,  0],
             [0,  0,  0,  0,   1,    0,  1,  0,  0],
             [0,  0, -1, -1,  -1,   -1, -1, -1,  0]],

            # . - x - [] X - . .
            [[0, -1, -1, -1, -99,   1, -1,  0,  0],
             [0,  0,  1,  0,   1,   0,  0,  0,  0],
             [0, -1, -1, -1,  -1,  -1, -1,  0,  0]],

            # . - - x [] X - - .
            [[0, -1, -1, -1,  -99,   1, -1, -1,  0],
             [0,  0,  0,  1,    1,   0,  0,  0,  0],
             [0, -1, -1, -1,   -1,  -1, -1, -1,  0]],

            # . . - - [] X x - -
            [[0,  0, -1, -1,  -99,   1, -1, -1, -1],
             [0,  0,  0,  0,    1,   0,  1,  0,  0],
             [0,  0, -1, -1,   -1,  -1, -1, -1, -1]],

            # . . . - [] X - x -
            [[0,  0,  0, -1,  -99,   1, -1, -1, -1],
             [0,  0,  0,  0,    1,   0,  0,  1,  0],
             [0,  0,  0, -1,   -1,  -1, -1, -1, -1]],

            # . . - x [] - X - .
            [[0,  0, -1, -1,  -99,  -1,  1, -1, 0],
             [0,  0,  0,  1,    1,   0,  0,  0, 0],
             [0,  0, -1, -1,   -1,  -1, -1, -1, 0]],

            # . . - - [] x X - -
            [[0,  0, -1, -1,  -99,  -1,  1, -1, -1],
             [0,  0,  0,  0,    1,   1,  0,  0,  0],
             [0,  0, -1, -1,   -1,  -1, -1, -1, -1]],

            # . . . - [] - X x -
            [[0,  0,  0, -1,  -99,  -1,  1, -1, -1],
             [0,  0,  0,  0,    1,   0,  0,  1,  0],
             [0,  0,  0, -1,   -1,  -1, -1, -1, -1]],

            # . . . - [] x - X -
            [[0,  0,  0, -1,  -99,  -1, -1,  1, -1],
             [0,  0,  0,  0,    1,   1,  0,  0,  0],
             [0,  0,  0, -1,   -1,  -1, -1, -1, -1]],

            # . . . - [] - x X -
            [[0,  0,  0, -1,  -99,  -1, -1,  1, -1],
             [0,  0,  0,  0,    1,   0,  1,  0,  0],
             [0,  0,  0, -1,   -1,  -1, -1, -1, -1]],
        ]

        filters, biases, weights = self.assemble_primary_filters()

        n_filters = len(biases)

        # Layer 1. Output: (curr/oth) x 4 directions x 32 patterns + 3 projectors => 259 channels per board
        self.detector = tf.keras.layers.Conv2D(
            filters=n_filters, kernel_size=(11, 11),
            kernel_initializer=tf.constant_initializer(filters),
            bias_initializer=tf.constant_initializer(biases),
            activation=tf.nn.relu,
            padding='same',
            input_shape=(board_size, board_size, 3))

        # Layer 2. Output: curr / other / boundary / inf_curr / inf_other
        weights = self.spread_weights(weights, n_filters)
        weights = np.rollaxis(np.array(weights), axis=-1)

        self.combine = tf.keras.layers.Conv2D(
            filters=5, kernel_size=(1, 1),
            kernel_initializer=tf.constant_initializer(weights),
            activation=tf.nn.tanh)

        # Forward-looking blocks, computing influence of influence
        #
        secondary_filters = self.assemble_secondary_filters()
        self.secondary_filters = secondary_filters
        n_symmetries = 4
        n_projectors = 3
        n_channels = 2
        n_filters_1 = len(self.secondary_patterns) * n_symmetries * n_channels
        n_all = n_filters_1 + n_projectors
        fw_weights = [1.] * n_all
        fw_weights = self.spread_weights_fw(fw_weights, n_all)
        fw_weights = np.rollaxis(np.array(fw_weights), axis=-1)

        self.forward_looking_1 = tf.keras.layers.Conv2D(
            filters=n_all, kernel_size=(9, 9),
            kernel_initializer=tf.constant_initializer(secondary_filters),
            bias_initializer=tf.constant_initializer([0., 0, 0.] + [-1.] * n_filters_1),
            padding='same',
            activation=tf.nn.relu
        )
        self.combine_fl_1 = tf.keras.layers.Conv2D(
            filters=5, kernel_size=(1, 1),
            kernel_initializer=tf.constant_initializer(fw_weights),
            activation=tf.nn.tanh
        )

    # ======================== End Forward-Looking Blocks

        # Simply add offensive and defensive advice
        self.finalize = tf.keras.layers.Conv2D(
            filters=1, kernel_size=(1, 1),
            kernel_initializer=tf.constant_initializer([0., 0., 0., 1., 1.])
        )

        self.peel = tf.keras.layers.Conv2D(
            filters=1, kernel_size=(3, 3),
            kernel_initializer=tf.constant_initializer([
                [[0., 0., 0.], [0., 1., 0.], [0., 0., 0.]]
            ]),
            bias_initializer=tf.constant_initializer(0.),
            trainable=False)


    def assemble_secondary_filters(self):

        secondary_patterns = self.secondary_patterns
        curr = np.stack([
            all_5xnxn(pattern, pov=0) for pattern in secondary_patterns
        ], axis=-1).reshape((9, 9, 5, -1))
        oth = np.stack([
            all_5xnxn(pattern, pov=1) for pattern in secondary_patterns
        ], axis=-1).reshape((9, 9, 5, -1))

        projectors = np.zeros((9, 9, 5, 3))
        projectors[4, 4, 0, 0] = 1.
        projectors[4, 4, 1, 1] = 1.
        projectors[4, 4, 2, 2] = 1.

        filters = np.concatenate([projectors, curr, oth], axis=3)

        return filters


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


    @staticmethod
    def spread_weights_fw(weights, n_channels):
        """
        distribute the weights across 5 different combiner 1x1xn_channels filters
        """
        # current stones projected
        c = [1] + [0] * (n_channels - 1)

        # other stones projected
        o = [0, 1] + [0] * (n_channels - 2)

        # boundary stones projected
        b = [0, 0, 1] + [0] * (n_channels - 3)

        # total influence of the current stones
        i = [0, 0, 0] + weights[3:]

        # total influence of the other stones
        j = [0, 0, 0] + weights[3:]

        return [c, o, b, i, j]


    def call(self, state):
        """
        :param state: state, representated as (n+2) x (n+2) x 3 board with boundary
        :return: the logit, clipped
        """
        # States are nxnx3
        state = np.reshape(state, [-1, self.input_size, self.input_size, 3]).astype(float)

        res1 = self.detector(state)
        res2 = self.combine(res1)
        res3 = self.forward_looking_1(res2)
        res4 = self.combine_fl_1(res3)
        res5 = self.finalize(res4)
        res6 = self.peel(res5)
        res7 = tf.clip_by_value(res6, 0, 999)  # anything beyond is overkill anyway.
        return res7

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

        return patterns


    def assemble_primary_filters(self):
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
