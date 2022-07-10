from typing import Callable

import numpy as np
import tensorflow as tf

from domoku.policies.radial import radial_3xnxn, all_5xnxn


class ForwardLookingLayer(tf.keras.layers.Layer, Callable):
    """
    A policy that doesn't miss any sure-win or must-defend
    """
    def __init__(self, board_size, boost=2.0, **kwargs):
        """
        :param board_size: length of the board including the boundary
        :param kwargs:
        """
        self.input_size = board_size + 2  # We include the boundary in the input
        self.boost = boost
        super().__init__(**kwargs)

        # Patterns to recognize combinations of potential stones with existing stones
        self.patterns = [

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

        projectors, curr, oth = self.assemble_filters()
        # n_symmetries = 4
        # n_projectors = 3
        # n_channels = 2
        # n_filters_1 = len(self.patterns) * n_symmetries * n_channels
        # n_all = n_filters_1 + n_projectors
        # fw_weights = [1.] * n_all
        # fw_weights = self.spread_weights(fw_weights, n_all)
        # fw_weights = np.rollaxis(np.array(fw_weights), axis=-1)

        self.projector = tf.keras.layers.Conv2D(
            filters=3, kernel_size=(9, 9),
            kernel_initializer=tf.constant_initializer(projectors),
            bias_initializer=tf.constant_initializer([0., 0, 0.]),
            padding='same'
        )

        self.fw_curr = tf.keras.layers.Conv2D(
            filters=72, kernel_size=(9, 9),
            kernel_initializer=tf.constant_initializer(curr),
            bias_initializer=tf.constant_initializer([-1.] * 72),
            padding='same',
            activation=tf.nn.relu
        )
        self.pool_curr = tf.keras.layers.MaxPooling3D(pool_size=(1, 1, 72))

        self.fw_oth = tf.keras.layers.Conv2D(
            filters=72, kernel_size=(9, 9),
            kernel_initializer=tf.constant_initializer(oth),
            bias_initializer=tf.constant_initializer([-1.] * 72),
            padding='same',
            activation=tf.nn.relu
        )
        self.pool_oth = tf.keras.layers.MaxPooling3D(pool_size=(1, 1, 72))


    def assemble_filters(self):

        curr = np.stack([
            all_5xnxn(pattern, pov=0) for pattern in self.patterns
        ], axis=-1).reshape((9, 9, 5, -1))
        oth = np.stack([
            all_5xnxn(pattern, pov=1) for pattern in self.patterns
        ], axis=-1).reshape((9, 9, 5, -1))

        projectors = np.zeros((9, 9, 5, 3))
        projectors[4, 4, 0, 0] = 1.
        projectors[4, 4, 1, 1] = 1.
        projectors[4, 4, 2, 2] = 1.

        filters = np.concatenate([projectors, curr, oth], axis=3)

        return projectors, curr, oth


    def call(self, signal):
        """
        :param signal: state, representated as (n+2) x (n+2) x 3 board with boundary
        :return: the logit, clipped
        """

        stones_n_boundary = self.projector(signal)

        curr_threat = self.fw_curr(tf.nn.tanh(self.boost * signal))
        curr_threat = tf.expand_dims(curr_threat, -1)
        curr_threat = self.pool_curr(curr_threat)
        curr_threat = curr_threat[:, :, :, :, 0]

        oth_threat = self.fw_oth(tf.nn.tanh(self.boost * signal))
        oth_threat = tf.expand_dims(oth_threat, -1)
        oth_threat = self.pool_oth(oth_threat)
        oth_threat = oth_threat[:, :, :, :, 0]

        curr_threat = curr_threat + tf.expand_dims(signal[:, :, :, 3], -1)
        oth_threat = oth_threat + tf.expand_dims(signal[:, :, :, 4], -1)
        res = tf.keras.layers.Concatenate()([stones_n_boundary, curr_threat, oth_threat])

        return res

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
