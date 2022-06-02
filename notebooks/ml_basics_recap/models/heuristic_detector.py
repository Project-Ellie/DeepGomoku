from tensorflow import keras
import tensorflow as tf
import numpy as np


class HeuristicDetector(keras.Model):

    def __init__(self, input_size, *args, **kwargs):
        """
        A simple 2-layer convolutional neural network for a Q-Function on a 2xNxN
        Gomoku board representation.
        :param input_size: The n of the square board
        """
        super().__init__(*args, **kwargs)
        self.input_size = input_size
        self.filters = self._create_filters()
        detector, combiner = self._create_layers()
        self.detector = detector
        self.combiner = combiner


    def call(self, x):
        """
        detect patterns and returns a number close to 1.0
        for positions featuring at least one line of 3 center
        :param x:
        :return:
        """
        y = self.detector(x)
        y = self.combiner(y)
        y = tf.reduce_max(y, axis=-1)
        y = tf.nn.tanh(5 * y)
        return y

    def _create_layers(self):
        detector = tf.keras.layers.Conv2D(
            name='detector',
            filters=4, kernel_size=(3, 3),
            kernel_initializer=tf.constant_initializer(self.filters),
            bias_initializer=tf.constant_initializer(-2),
            padding='same',
            activation=tf.nn.relu,
            input_shape=(self.input_size, self.input_size, 2))

        combiner = tf.keras.layers.Conv2D(
            name='combiner',
            filters=1, kernel_size=(1, 1),
            kernel_initializer=tf.constant_initializer([1., 1., 1., 1.]),
            bias_initializer=tf.constant_initializer(0))

        return detector, combiner


    @staticmethod
    def _create_filters():
        diag1 = np.diag([1., 1., 1.])
        zeros = np.zeros([3, 3])
        diag1 = np.stack([diag1, zeros], axis=0)
        diag2 = np.diag([1., 1., 1.])[::-1, :]
        diag2 = np.stack([diag2, zeros], axis=0)
        hor = np.stack([zeros, zeros], axis=0)
        hor[0, 1, :] = 1.
        ver = np.stack([zeros, zeros], axis=0)
        ver[0, :, 1] = 1.
        filters = np.stack([ver, diag1, hor, diag2], axis=-1)
        filters = np.rollaxis(filters, 0, 3)
        return filters
