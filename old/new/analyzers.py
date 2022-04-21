import numpy as np
import tensorflow as tf
from old.new.wgomoku import GomokuBoard, create_sample


class Analyzer:

    def __init__(self, n):
        self.n = n
        empty = np.zeros([5, 5], dtype=float)
        diag = np.eye(5, dtype=float)
        filters = np.array([
            [empty, diag[::-1]],
            [diag[::-1], empty]])
        self.filters = np.rollaxis(filters, 0, 3)

        kernel_init = tf.constant_initializer(self.filters)
        bias_init = tf.constant_initializer(-4.)

        self.model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(filters=2, kernel_size=(5, 5),
                                   kernel_initializer=kernel_init, bias_initializer=bias_init,
                                   activation=tf.nn.relu, input_shape=(6, 6, 2,)), ])


    def detect_five(self, board: GomokuBoard, color: int):
        """
        Detects whether the given board features a line of 5 of the given color
        :param board: the board to be analyzed
        :param color: the color: BLACK (0) or WHITE (1)
        """
        sample = create_sample(board.stones, board.N, color, borders=False)
        sample = np.reshape(sample, [-1, self.n, self.n, 2])
        recognized = self.model(sample)
        detected = np.squeeze(recognized.numpy())
        return (detected == 1).any()
