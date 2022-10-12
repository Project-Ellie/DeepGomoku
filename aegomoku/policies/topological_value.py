import numpy as np
import tensorflow as tf

from aegomoku.interfaces import Adviser
from aegomoku.policies.radial import all_3xnxn


class TopologicalValuePolicy(tf.keras.Model, Adviser):
    """
    computes a value af each position as a pseudo-euclidean sum of the values of all that position's lines of five
    The value of a line of five is the number of stones of a color if that color is exclusive
    on the respective line, zero otherwise.
    parallel line values add like v = sum(v_s ** 6) ** (1/6); s from range(5) - the shifts from ____. to .____
    values of the 4 different directions add like v = (sum(v_d ** 5) ** (1/5); d from range(4) - the directions
    """


    def __init__(self, board_size, kappa_s: float = 6.0, kappa_d: float = 5.0,
                 policy_stretch: float = 2.0, value_stretch: float = 1 / 32.,
                 advice_cutoff: float = .01, noise_reduction: float = 1.1,
                 percent_secondary: float = 34, min_secondary: int = 5,
                 value_gauge: float = 0.1):
        """
        :param kappa_s: exponent of the pseudo-euclidean sum of parallel lines-of-five
        :param kappa_d: exponent of the pseudo-euclidean sum of different directions
        :param policy_stretch: A factor, applied to logits before they're fed into the softwmax
        :param value_stretch: A factor, applied to the raw value before it's fed into the tanh
        :param value_gauge: factor applied to the value after tanh has been applied
        :param noise_reduction: factor applied to lowest prob before subtracting from signal
        :param percent_secondary: percentage of not-so-popular moves taken into advisable action. Our 'Dirichlet' noise.
        """
        super().__init__()
        self.kappa_s = kappa_s
        self.kappa_d = kappa_d
        self.policy_stretch = policy_stretch
        self.value_stretch = value_stretch
        self.advice_cutoff = advice_cutoff
        self.noise_reduction = noise_reduction
        self.percent_secondary = percent_secondary
        self.min_secondary = min_secondary
        self.value_gauge = value_gauge
        self.board_size = board_size

        self.raw_patterns = [
            [[0, 0, 0, 0, -5, 1, 1, 1, 1], [0, 0, 0, 0, -5, -5, -5, -5, -5], 0],
            [[0, 0, 0, 1, -5, 1, 1, 1, 0], [0, 0, 0, -5, -5, -5, -5, -5, 0], 0],
            [[0, 0, 1, 1, -5, 1, 1, 0, 0], [0, 0, -5, -5, -5, -5, -5, 0, 0], 0],
            [[0, 1, 1, 1, -5, 1, 0, 0, 0], [0, -5, -5, -5, -5, -5, 0, 0, 0], 0],
            [[1, 1, 1, 1, -5, 0, 0, 0, 0], [-5, -5, -5, -5, -5, 0, 0, 0, 0], 0]]

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

        self.p_v = tf.keras.layers.Conv2D(
            name="sum_d",
            filters=2,
            kernel_size=1,
            kernel_initializer=tf.constant_initializer([
                    [1, 1],
                    [1, -1]
                ]),
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
        y = tf.math.pow(y, 1 / self.kappa_d)
        y = self.p_v(y)
        p, v = y[:, :, :, 0], y[:, :, :, 1]
        p = self.peel(tf.expand_dims(p, -1))
        v = self.peel(tf.expand_dims(v, -1))

        probs = tf.reshape(p, (-1, ))
        probs = tf.keras.layers.Softmax()(self.policy_stretch * probs)
        value = self.value_gauge * tf.nn.tanh(self.value_stretch * tf.math.reduce_sum(v))
        return probs, value

    def get_advisable_actions(self, state, cut_off=None, reduction=None,
                              percent_secondary=None, min_secondary=None):
        """
        :param state: nxnx3 representation of a go board
        :param cut_off: override advice_cutoff
        :param reduction: override noise reduction
        :param percent_secondary: override the percentage of low quality moves to include.
        :param min_secondary: override the minumum of low quality moves to include
        :return: List of integers representing the avisable moves
        """
        cut_off = cut_off if cut_off is not None else self.advice_cutoff
        # reduction = reduction if reduction is not None else self.noise_reduction
        percent_secondary = percent_secondary if percent_secondary is not None else self.percent_secondary
        min_secondary = min_secondary if min_secondary is not None else self.min_secondary

        probs, _ = self.evaluate(np.squeeze(state))
        max_prob = np.max(probs, axis=None)
        probs = np.squeeze(probs)

        # noise = np.min(probs, axis=None)
        # probs = tf.keras.activations.relu(probs - reduction * noise)

        threshold = max_prob * cut_off

        advisable = np.where(probs > threshold, probs, 0.)
        advisable = [int(n) for n in advisable.nonzero()[0]]

        # Occasionally consider an underdog - you never know...;-)
        underdogs = np.where(probs <= threshold, probs, 0.)
        secondary = [int(n) for n in underdogs.nonzero()[0]]

        # stretch to increase the softmax focus
        subset = probs[secondary]

        # rule out the lowest of the lower quality moves

        if len(subset > 0):
            min_p = min(subset)
            subset = (subset - min_p) * 10.0

            # probability on the subset for random choice
            secondary_probs = tf.nn.softmax(subset).numpy()

            # normalize potential numerical errors
            sum_probs = sum(secondary_probs)
            secondary_probs /= sum_probs

            # take a certain percentage of sub-prime choices, but at least two.
            num_additional = max(min_secondary, int((len(advisable) * percent_secondary / 100)))
            selected_secondary = np.random.choice(secondary, size=num_additional, p=secondary_probs)

            advisable += list(selected_secondary)

        return advisable

    def evaluate(self, state):
        state = np.expand_dims(state, 0).astype(float)
        probs, value = self.call(state)
        return np.squeeze(probs), value.numpy()

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
