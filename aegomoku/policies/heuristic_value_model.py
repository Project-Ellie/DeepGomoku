import numpy as np
import tensorflow as tf

from aegomoku.policies.heuristic_advice import HeuristicValueParams
from aegomoku.policies.radial import all_3xnxn


class HeuristicValueModel(tf.keras.Model):
    """
    computes a value af each position as a pseudo-euclidean sum of the values of all that position's lines of five
    The value of a line of five is the number of stones of a color if that color is exclusive
    on the respective line, zero otherwise.
    parallel line values add like v = sum(v_s ** 6) ** (1/6); s from range(5) - the shifts from ____. to .____
    values of the 4 different directions add like v = (sum(v_d ** 5) ** (1/5); d from range(4) - the directions
    """


    def __init__(self, params: HeuristicValueParams):
        super().__init__()
        self.kappa_s = params.kappa_s
        self.kappa_d = params.kappa_d
        self.value_stretch = params.value_stretch
        self.value_gauge = params.value_gauge
        self.current_advantage = params.current_advantage
        self.bias = params.bias
        self.board_size = params.board_size

        self.raw_patterns = [
            [[0, 0, 0, 0, -5, 1, 1, 1, 1],
             [0, 0, 0, 0, -5, -5, -5, -5, -5], self.bias],

            [[0, 0, 0, 1, -5, 1, 1, 1, 0],
             [0, 0, 0, -5, -5, -5, -5, -5, 0], self.bias],

            [[0, 0, 1, 1, -5, 1, 1, 0, 0],
             [0, 0, -5, -5, -5, -5, -5, 0, 0], self.bias],

            [[0, 1, 1, 1, -5, 1, 0, 0, 0],
             [0, -5, -5, -5, -5, -5, 0, 0, 0], self.bias],

            [[1, 1, 1, 1, -5, 0, 0, 0, 0],
             [-5, -5, -5, -5, -5, 0, 0, 0, 0], self.bias]]

        self.num_filters = len(self.raw_patterns) * 8

        self.patterns = self.select_patterns()

        filters, biases = self.assemble_filters()

        self.detector = tf.keras.layers.Conv2D(
            name="heuristic_value",
            filters=self.num_filters,
            kernel_size=(9, 9),
            kernel_initializer=tf.constant_initializer(filters),
            bias_initializer=tf.constant_initializer(biases),
            activation=tf.nn.relu,
            padding='same',
            trainable=False)

        sum_s_filters = np.array([
            5 * [1, 0, 0, 0, 0, 0, 0, 0],
            5 * [0, 1, 0, 0, 0, 0, 0, 0],
            5 * [0, 0, 1, 0, 0, 0, 0, 0],
            5 * [0, 0, 0, 1, 0, 0, 0, 0],
            5 * [0, 0, 0, 0, 1, 0, 0, 0],
            5 * [0, 0, 0, 0, 0, 1, 0, 0],
            5 * [0, 0, 0, 0, 0, 0, 1, 0],
            5 * [0, 0, 0, 0, 0, 0, 0, 1],
            ]).T

        self.sum_s = tf.keras.layers.Conv2D(
            name="sum_s",
            filters=8,
            kernel_size=1,
            kernel_initializer=tf.constant_initializer(sum_s_filters),
            bias_initializer=tf.constant_initializer(0.),
            padding='same',
            trainable=False)

        sum_d_filters = np.array([
            [1, 1, 1, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 1, 1, 1]
        ]).T

        self.sum_d = tf.keras.layers.Conv2D(
            name="sum_d",
            filters=2,
            kernel_size=1,
            kernel_initializer=tf.constant_initializer(sum_d_filters),
            bias_initializer=tf.constant_initializer(0.),
            padding='same',
            trainable=False)

        self.diff = tf.keras.layers.Conv2D(
            name="diff",
            filters=1,
            kernel_size=1,
            kernel_initializer=tf.constant_initializer([1 + self.current_advantage, -1]),
            bias_initializer=tf.constant_initializer(0.),
            padding='same',
            trainable=False)

        self.peel = tf.keras.layers.Conv2D(
            filters=1, kernel_size=(3, 3),
            kernel_initializer=tf.constant_initializer([
                [[0., 0., 0.], [0., 1., 0.], [0., 0., 0.]]
            ]),
            bias_initializer=tf.constant_initializer(0.),
            trainable=False)


    def call(self, state):
        y = self.detector(state)
        y = tf.math.pow(y, self.kappa_s)
        y = self.sum_s(y)
        y = tf.math.pow(y, self.kappa_d/self.kappa_s)
        y = self.sum_d(y)

        v = self.peel(self.diff(y))

        value = self.value_gauge * tf.nn.tanh(self.value_stretch * tf.math.reduce_sum(v))
        return value

    def evaluate(self, state):
        state = np.expand_dims(state, 0).astype(float)
        value = self.call(state)
        return value.numpy()

    def advise(self, state):
        """
        :param state: nxnx3 representation of a go board
        :returns: a selection of move probabilities: a subset of the policy, renormalized
        """
        bits = np.zeros([self.board_size * self.board_size], dtype=np.uint)
        advisable = self.get_advisable_actions(np.expand_dims(state, 0).astype(float))
        bits[advisable] = 1
        p, _ = self.evaluate(state)
        probs = bits * p
        total = np.sum(probs)
        return probs / total


    def select_patterns(self):
        return [
            [[offense, defense, defense], bias]
            if channel == 0
            else [[defense, offense, defense], bias]
            for offense, defense, bias in self.raw_patterns
            for channel in [0, 1]
        ]


    def assemble_filters(self):
        """
        Considering the boundary stones just as good a defense as one of the opponent's stone.
        Boundary stones are placed on the periphery of the 3rd channel
        """
        the_filters = np.stack([
            all_3xnxn(pattern[0])
            for pattern in self.patterns], axis=3)

        the_biases = []
        for pattern in self.patterns:
            the_biases += 4 * [pattern[1]]

        return np.reshape(the_filters, (9, 9, 3, -1)), the_biases

#%%
