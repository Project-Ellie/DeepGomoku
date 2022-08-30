import numpy as np
import tensorflow as tf
from aegomoku.interfaces import TerminalDetector, Adviser
from aegomoku.policies.forward_looking import ForwardLookingLayer
from aegomoku.policies.naive_infuence import NaiveInfluenceLayer
from aegomoku.policies.primary_detector import PrimaryDetector


class HeuristicPolicy(tf.keras.Model, Adviser, TerminalDetector):

    def __init__(self, board_size: int, cut_off: float = 0.8):
        """
        :param board_size:
        :param cut_off: discard advisable actions with probability less than cut_off * highest
        """

        self.board_size = board_size
        self.cut_off = cut_off
        super().__init__()

        self.peel = tf.keras.layers.Conv2D(
            filters=1, kernel_size=(3, 3),
            kernel_initializer=tf.constant_initializer([
                [[0., 0., 0.], [0., 1., 0.], [0., 0., 0.]]
            ]),
            bias_initializer=tf.constant_initializer(0.),
            trainable=False)

        self.detector = PrimaryDetector(board_size)
        self.fwll = ForwardLookingLayer(board_size)
        self.influence = NaiveInfluenceLayer(board_size)
        self.influence_weight = 0.01


    def get_winner(self, state):
        c, o = self.logits(state)
        c = np.max(c, axis=None)
        o = np.max(o, axis=None)
        if c > 9000:
            return 0
        elif o > 3000:
            return 1
        else:
            return None

    def get_advisable_actions(self, state):
        """
        :param: state: batch of a single state. 1 x N x N x 3
        :return: a list of integer move representations with probabilities close enough to the maximum (see: cut_off)
        """
        probs, _ = self.call(state)
        max_prob = np.max(probs, axis=None)
        probs = probs.reshape(self.board_size * self.board_size)
        advisable = np.where(probs > max_prob * self.cut_off, probs, 0.)
        return [int(n) for n in advisable.nonzero()[0]]


    def logits(self, s):
        raw = self.fwll(self.detector(s)) + self.influence(s) * self.influence_weight # noqa
        logits_c = self.squeeze_and_peel(raw, 3)
        logits_o = self.squeeze_and_peel(raw, 4)
        return logits_c, logits_o


    def squeeze_and_peel(self, raw, layer):
        return np.squeeze(self.peel(tf.expand_dims(raw[:, :, :, layer], -1)))

    def call(self, s):
        logits_c, logits_o = self.logits(s)
        eps = 1e-8
        logits = logits_c + logits_o
        mx = np.max(logits, axis=None)
        mn = np.min(logits, axis=None)
        if mx == mn:
            rescaled = np.array([0.] * self.board_size * self.board_size)
        else:
            rescaled = np.log((logits - mn) / (mx - mn) * np.e + eps)
            rescaled = rescaled.reshape(self.board_size * self.board_size)
        probs = tf.nn.softmax(rescaled)
        probs = np.reshape(probs, [self.board_size, self.board_size])
        value = tf.nn.tanh(np.sum(logits_c - logits_o, axis=None) / 600.)

        return probs, value


    def evaluate(self, state):
        state = np.expand_dims(state, 0).astype(float)
        pi, v = self.call(state)
        pi = np.reshape(pi, self.board_size * self.board_size)
        return pi, v
