import dataclasses
from typing import List

import numpy as np
import tensorflow as tf

from aegomoku.interfaces import Adviser
from aegomoku.policies.threat_detector import ThreatDetector


@dataclasses.dataclass
class HeuristicPolicyParams:
    board_size: int
    advice_threshold: float = .3
    criticalities: List[float] = dataclasses.field(default_factory=lambda: [999, 333, 99, 33, 2, 2, 2, 2, 1, 1])
    percent_secondary: float = 0.
    min_secondary: float = 0
    aggregator_power: int = 2.5


@dataclasses.dataclass
class HeuristicValueParams:
    """
    :param kappa_s: exponent of the pseudo-euclidean sum of parallel lines-of-five
    :param kappa_d: exponent of the pseudo-euclidean sum of different directions
    :param value_stretch: A factor, applied to the raw value before it's fed into the tanh
    :param value_gauge: factor applied to the value after tanh has been applied
    :param current_advantage: The additional value (as fraction of total) for the current player
    :param bias: added to the number of stones counted. A good value is -0.5 to bias against single stones
    """
    board_size: int
    kappa_s: int = 5
    kappa_d: int = 2
    value_stretch: float = .02
    value_gauge: float = 1.0
    current_advantage: float = .1
    bias: float = -0.5


class HeuristicAdviser(tf.keras.Model, Adviser):

    def __init__(self, params: HeuristicPolicyParams, value_model):
        super().__init__()
        self.params = params
        self.board_size = params.board_size
        self.detector = ThreatDetector(self.params.board_size)
        self.value_model = value_model

        kernel = np.diag(params.criticalities)

        self.criticality = tf.keras.layers.Conv2D(
            name='weights',
            filters=10, kernel_size=(1, 1),
            trainable=False,
            kernel_initializer=tf.constant_initializer(kernel))

        self.sum_channels = tf.keras.layers.Conv2D(
            name='heuristic_priority',
            filters=1, kernel_size=(1, 1),
            trainable=False,
            kernel_initializer=tf.constant_initializer([1.] * 10))

        self.flatten = tf.keras.layers.Flatten()

        self.peel = tf.keras.layers.Conv2D(
            filters=1, kernel_size=(3, 3),
            kernel_initializer=tf.constant_initializer([
                [[0., 0., 0.], [0., 1., 0.], [0., 0., 0.]]
            ]),
            bias_initializer=tf.constant_initializer(0.),
            trainable=False)

    def call(self, state):
        output = self.detector.call(state)
        crit = self.criticality(output)

        crit = tf.pow(crit, self.params.aggregator_power)
        crit = self.sum_channels(crit)
        crit = tf.pow(crit, 1./self.params.aggregator_power)

        crit = self.peel(crit)
        crit = self.flatten(crit)

        softmax = tf.keras.layers.Softmax()(crit)

        value = self.value_model.call(state)
        return softmax, value

    def get_advisable_actions(self, state, cut_off=None,
                              percent_secondary=None, min_secondary=None):
        """
        :param state: nxnx3 representation of a go board
        :param cut_off: override advice_cutoff
        :param percent_secondary: override the percentage of low quality moves to include.
        :param min_secondary: override the minumum of low quality moves to include
        :return: List of integers representing the avisable moves
        """
        cut_off = cut_off if cut_off is not None else self.params.advice_threshold

        percent_secondary = percent_secondary if percent_secondary is not None else self.params.percent_secondary
        min_secondary = min_secondary if min_secondary is not None else self.params.min_secondary

        probs, _ = self.evaluate(state)

        min_prob = np.min(probs, axis=None)
        probs = probs - min_prob

        max_prob = np.max(probs, axis=None)

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
