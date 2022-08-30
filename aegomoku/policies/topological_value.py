import numpy as np
import tensorflow as tf

from aegomoku.interfaces import Adviser, TerminalDetector
from aegomoku.policies.radial import all_3xnxn


class TopologicalValuePolicy(tf.keras.Model, Adviser, TerminalDetector):
    """
    computes a value af each position as a topological some of the values of all that position's lines of five
    The value of a line of five is the number of stones of a color if that color is exclusive
    on the respective line, zero otherwise.
    parallel line values add like v = sum(v_s ** 6) ** (1/6); s from range(5) - the shifts from ____. to .____
    values of the 4 different directions add like v = (sum(v_d ** 5) ** (1/5); d from range(4) - the directions
    """


    def __init__(self, kappa_s: float = 6.0, kappa_d: float = 5.0,
                 policy_stretch: float = 2.0, value_stretch: float = 1 / 32.,
                 advice_cutoff: float = .01, noise_reduction: float = 1.1,
                 percent_secondary: float = 34, min_secondary: int = 5,
                 value_gauge: float = 0.2):
        """
        :param kappa_s: exponent of the pseudo-euclidean sum of parallel lines-of-five
        :param kappa_d: exponent of the pseudo-euclidean sum of different directions
        :param policy_stretch: A factor, applied to logits before they're fed into the softwmax
        :param value_stretch: A factor, applied to the raw value before it's fed into the tanh
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
        self.num_filters = 48

        self.raw_patterns = [
            [[0, 0, 0, 0, -5, 1, 1, 1, 1], [0, 0, 0, 0, -5, -5, -5, -5, -5], 0],
            [[0, 0, 0, 1, -5, 1, 1, 1, 0], [0, 0, 0, -5, -5, -5, -5, -5, 0], 0],
            [[0, 0, 1, 1, -5, 1, 1, 0, 0], [0, 0, -5, -5, -5, -5, -5, 0, 0], 0],
            [[0, 1, 1, 1, -5, 1, 0, 0, 0], [0, -5, -5, -5, -5, -5, 0, 0, 0], 0],
            [[1, 1, 1, 1, -5, 0, 0, 0, 0], [-5, -5, -5, -5, -5, 0, 0, 0, 0], 0],

            # Terminal detection - note that this won't detect overlines!
            [[0, -5, 1, 1, 1, 1, 1, -5, 0], [0, 0, -5, -5, -5, -5, -5, 0, 0], -4]]

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

        self.peel = tf.keras.layers.Conv2D(
            filters=1, kernel_size=(3, 3),
            kernel_initializer=tf.constant_initializer([
                [[0., 0., 0.], [0., 1., 0.], [0., 0., 0.]]
            ]),
            bias_initializer=tf.constant_initializer(0.),
            trainable=False)


    def get_advisable_actions(self, state, cut_off=None, reduction=None):
        """
        :param state: nxnx3 representation of a go board
        :param cut_off: override advice_cutoff
        :param reduction: override noise reduction
        :return:
        """
        cut_off = cut_off if cut_off is not None else self.advice_cutoff
        reduction = reduction if reduction is not None else self.noise_reduction

        probs, _ = self.evaluate(np.squeeze(state))
        max_prob = np.max(probs, axis=None)
        probs = np.squeeze(probs)

        noise = np.min(probs, axis=None)
        probs = tf.keras.activations.relu(probs - reduction * noise)

        threshold = max_prob * cut_off

        advisable = np.where(probs > threshold, probs, 0.)
        advisable = [int(n) for n in advisable.nonzero()[0]]

        # Occasionally consider an underdog - you never know...;-)
        underdogs = np.where(probs <= threshold, probs, 0.)
        secondary = [int(n) for n in underdogs.nonzero()[0]]

        # stretch to increase the softmax focus
        subset = probs.numpy()[secondary]

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
            num_additional = max(self.min_secondary, int((len(advisable) * self.percent_secondary / 100)))
            selected_secondary = np.random.choice(secondary, size=num_additional, p=secondary_probs)

            advisable += list(selected_secondary)

        return advisable


    def call(self, state):
        field_current, ftc = self.field(state, channel=0)
        field_other, fto = self.field(state, channel=1)
        logits = field_current + field_other
        logits = np.reshape(logits, -1)

        probs = tf.keras.layers.Softmax()(self.policy_stretch * logits)
        value = tf.nn.tanh(self.value_stretch * self.value(state))
        return probs, value

    def evaluate(self, state):
        state = np.expand_dims(state, 0).astype(float)
        probs, value = self.call(state)
        return probs.numpy(), value.numpy()


    def get_winner(self, state):
        _, ftc = self.field(state, 0)
        _, fto = self.field(state, 1)
        if np.sum(ftc, axis=None) > 0 or np.sum(fto, axis=None) > 0:
            return (np.sum(state[:, :, :2]) + 1) % 2


    def value(self, state):
        """
        :return: the sum of current minus other potential positions
        """
        current_value, other_value = [  # for current and other channel
            np.sum(  # over the entire board positions
                self.field(state, channel)[0]
            )
            for channel in range(2)
        ]

        return self.value_gauge * (current_value - other_value)


    def field(self, state, channel):
        """
        Compute a pair of two value fields
        :param state:
        :param channel:
        :return: The field of 'potential' for the given channel and the field of terminal values
        """

        # feature images of the 48 filters: 4 directions times 5 shifts times 2 channels (current + other) + terminals
        counts = self.detector(state)
        features = [np.squeeze(self.peel(tf.expand_dims(counts[:, :, :, feature_no], -1)))
                    for feature_no in range(self.num_filters)]

        # soft values
        f = sum([  # over the values of different directions
            sum([  # over the values of parallel lines
                features[8 * shift + 4 * channel + direction] ** self.kappa_s
                for shift in range(5)
            ]) ** (self.kappa_d / self.kappa_s)
            for direction in range(4)
        ]) ** (1 / self.kappa_d)

        # terminal values
        f_t = sum([  # over the values of different directions
            features[40 + 4 * channel + direction]
            for direction in range(4)
        ])

        return f, f_t


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
