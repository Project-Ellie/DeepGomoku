import numpy as np

import tensorflow as tf
from domoku.policies.radial import all_2xnxn

# Criticality Categories
TERMINAL = 0
FIN_IN_2 = 1

CRITICALITIES = [
    TERMINAL, FIN_IN_2
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
    def __init__(self, input_size, **kwargs):

        self.input_size = input_size
        super().__init__(**kwargs)

        self.patterns = [
            # win-in-1 patterns
            [
                [[1, 1, 1, 1, -1, 0, 0, 0, 0], [0, 0, 0, 0, -1, 0, 0, 0, 0], -3, [99, 50]],
                [[0, 1, 1, 1, -1, 1, 0, 0, 0], [0, 0, 0, 0, -1, 0, 0, 0, 0], -3, [99, 50]],
                [[0, 0, 1, 1, -1, 1, 1, 0, 0], [0, 0, 0, 0, -1, 0, 0, 0, 0], -3, [99, 50]],
                [[0, 0, 0, 1, -1, 1, 1, 1, 0], [0, 0, 0, 0, -1, 0, 0, 0, 0], -3, [99, 50]],
                [[0, 0, 0, 0, -1, 1, 1, 1, 1], [0, 0, 0, 0, -1, 0, 0, 0, 0], -3, [99, 50]],
            ],
            # win-in-2 patterns
            [
                [[-1, 1, 1, 1, -1, -1, 0, 0, 0], [-1, 0, 0, 0, -1, -1, 0, 0, 0], -2, [10, 5]],
                [[0, -1, 1, 1, -1, 1, -1, 0, 0], [0, -1, 0, 0, -1, 0, -1, 0, 0], -2, [10, 5]],
                [[0, 0, -1, 1, -1, 1, 1, -1, 0], [0, 0, -1, 0, -1, 0, 0, -1, 0], -2, [10, 5]],
                [[0, 0, 0, -1, -1, 1, 1, 1, -1], [0, 0, 0, -1, -1, 0, 0, 0, -1], -2, [10, 5]],
            ],
            # open 3 patterns - not so critical
            [
                [[-1,  1, -1,  1, -1, -1,  0,  0,  0], [-1, -1, -1, -1, -1, -1,  0,  0,  0], -1, [5, 1]],
                [[+0, -1,  1, -1, -1,  1, -1,  0,  0], [+0, -1, -1, -1, -1, -1, -1,  0,  0], -1, [5, 1]],
                [[+0,  0, -1,  1, -1, -1,  1, -1,  0], [+0,  0, -1, -1, -1, -1, -1, -1,  0], -1, [5, 1]],
                [[+0,  0,  0, -1, -1,  1, -1,  1, -1], [+0,  0,  0, -1, -1, -1, -1, -1, -1], -1, [5, 1]],

                [[-1,  1,  1, -1, -1, -1,  0,  0,  0], [-1, -1, -1, -1, -1, -1,  0,  0,  0], -1, [5, 1]],
                [[+0, -1,  1,  1, -1, -1, -1,  0,  0], [+0, -1, -1, -1, -1, -1, -1,  0,  0], -1, [5, 1]],
                [[+0,  0, -1,  1, -1,  1, -1, -1,  0], [+0,  0, -1, -1, -1, -1, -1, -1,  0], -1, [5, 1]],
                [[+0,  0,  0, -1, -1,  1,  1, -1, -1], [+0,  0,  0, -1, -1, -1, -1, -1, -1], -1, [5, 1]],
            ]
        ]

        filters, biases, weights = self.assemble_filters()

        self.detector = tf.keras.layers.Conv2D(
            filters=len(biases), kernel_size=(9, 9),
            kernel_initializer=tf.constant_initializer(filters),
            bias_initializer=tf.constant_initializer(biases),
            activation=tf.nn.relu,
            padding='same',
            input_shape=(input_size, input_size, 2))

        self.combine = tf.keras.layers.Conv2D(
            filters=1, kernel_size=(1, 1),
            kernel_initializer=tf.constant_initializer(weights))

    def call(self, sample):
        # Allow other representations here by ignoring their additional channels
        if sample.shape[-1] == 4:
            sample = sample[:, :, :2]

        sample = np.reshape(sample, [-1, self.input_size, self.input_size, 2])
        res = self.combine(self.detector(sample))
        return res

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
        reshaped = np.reshape(stacked, (9, 9, 2, 4 * np.shape(patterns)[0]))

        return reshaped, biases, weights
