import numpy as np
from keras import models

from aegomoku.interfaces import Adviser, PolicyParams
from aegomoku.utils import expand


class PolicyAdviser(Adviser):

    def __init__(self, model: models.Model, params: PolicyParams):
        self.model = model
        self.params = params
        self.board_size = params.board_size

    def get_advisable_actions(self, state):
        """
        :param state: nxnx3 representation of a go board
        :return:
        """
        state = expand(state)
        probs, _ = self.model(state)
        max_prob = np.max(probs, axis=None)
        probs = np.squeeze(probs)
        advisable = np.where(probs > max_prob * self.params.advice_cutoff, probs, 0.)

        return advisable.nonzero()[0].astype(int)
        # return [int(n) for n in advisable.nonzero()[0]]

    def advise(self, state):
        """
        :param state: nxnx3 representation of a go board
        :returns: a selection of move probabilities: a subset of the policy, renormalized
        """
        bits = np.zeros([self.board_size * self.board_size], dtype=np.uint)
        advisable = self.get_advisable_actions(state)
        bits[advisable] = 1
        p, _ = self.evaluate(state)
        probs = bits * p
        total = np.sum(probs)
        return probs / total


    def evaluate(self, state):
        """
        :param state: nxnx3 representation of a go board
        :return:
        """
        inputs = np.expand_dims(state, 0).astype(float)
        p, v = self.model(inputs)
        return np.squeeze(p), np.squeeze(v)
