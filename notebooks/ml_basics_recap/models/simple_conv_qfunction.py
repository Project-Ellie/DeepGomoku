from tensorflow import keras
import tensorflow as tf


class SimpleConvQFunction(keras.Model):
    def __init__(self, input_size, n_layers, n_filters, *args, **kwargs):
        """
        A simple 2-layer convolutional neural network for a Q-Function on a 2xNxN
        Gomoku board representation.
        :param input_size: The size of the square board
        """
        super().__init__(*args, **kwargs)
        self.input_size = input_size
        self.n_layers = n_layers
        self.n_filters = n_filters
        conv_chain, conv_head = self._create_layers()
        self.conv_chain = conv_chain
        self.conv_head = conv_head


    def call(self, x):
        """
        :param x: a (B, 1, N, N, 2) - shaped batch of B samples of NxXx2 representations
        :return: a (B, 1, 7, 7) tensor representing the Q-Function's output [-1, 1] for possible Actions
        """
        y = x
        for conv in self.conv_chain:
            y = conv(y)
        y = self.conv_head(y)
        y = tf.reduce_max(y, axis=4)
        return y


    def _create_layers(self):
        conv_chain = [
            tf.keras.layers.Conv2D(
                filters=self.n_filters, kernel_size=(3, 3),
                kernel_initializer=tf.random_normal_initializer(),
                bias_initializer=tf.random_normal_initializer(),
                padding='same',
                activation=tf.nn.relu,
                input_shape=(self.input_size, self.input_size, 2))
            for _ in range(self.n_layers)]

        conv_head = tf.keras.layers.Conv2D(
            filters=self.n_filters, kernel_size=(3, 3),
            kernel_initializer=tf.random_normal_initializer(),
            bias_initializer=tf.random_normal_initializer(),
            padding='same',
            activation=tf.nn.tanh,
            input_shape=(self.input_size, self.input_size, 2))

        return conv_chain, conv_head
