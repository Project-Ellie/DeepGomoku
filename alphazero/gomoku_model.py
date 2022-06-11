import numpy as np
import tensorflow as tf


class GomokuModel(tf.keras.Model):
    """
    A naive model just to start with something
    """

    def __init__(self, input_size: int, kernel_size):
        super().__init__()
        self.input_size = input_size
        self.kernel_size = kernel_size

        first, pot, agg, peel = self.create_model()
        self.first = first
        self.potentials = pot
        self.aggregate = agg
        self.peel = peel


    def call(self, sample):

        # add two more channels filled with zeros. They'll be carrying the 'influence' of the surrounding stones.
        # That allows for arbitrarily deep chaining within our architecture

        y = self.first(sample)
        for potential in self.potentials:
            y = potential(y)
        soft = self.peel(self.aggregate(y))

        value = tf.reduce_max(soft)
        pi = tf.nn.softmax(tf.keras.layers.Flatten()(soft))

        return pi, value


    def create_model(self):

        # Compute the current player's total potential, can be arbitrarily repeated
        # to create some forward-looking capabilities
        first = tf.keras.layers.Conv2D(
            filters=32, kernel_size=self.kernel_size,
            kernel_initializer=tf.random_normal_initializer(),
            bias_initializer=tf.random_normal_initializer(),
            activation=tf.nn.relu,
            padding='same',
            input_shape=(self.input_size, self.input_size, 3))

        potentials = [
            tf.keras.layers.Conv2D(
                name=f'Potential_{i}',
                filters=32, kernel_size=self.kernel_size,
                kernel_initializer=tf.random_normal_initializer(),
                bias_initializer=tf.random_normal_initializer(),
                activation=tf.nn.relu,
                padding='same',
                input_shape=(self.input_size, self.input_size, 5))
            for i in range(5)
            ]

        aggregate = tf.keras.layers.Conv2D(
            filters=1, kernel_size=1,
            kernel_initializer=tf.random_normal_initializer(),
            bias_initializer=tf.random_normal_initializer(),
            activation=tf.nn.relu,
            padding='same',
            input_shape=(self.input_size-1, self.input_size-1, 5))

        # 'peel' off the boundary and provide Q semantics with tanh
        peel = tf.keras.layers.Conv2D(
            filters=1, kernel_size=(3, 3),
            kernel_initializer=tf.constant_initializer([[0., 0., 0.], [0., 1., 0.], [0., 0., 0.]]),
            bias_initializer=tf.constant_initializer(0.),
            activation=tf.nn.tanh,
            trainable=False)

        return first, potentials, aggregate, peel
