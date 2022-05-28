from typing import Optional

import numpy as np
from pydantic import BaseModel
import tensorflow as tf

from domoku.constants import Move
from domoku.interfaces import AbstractGanglion
from domoku.policies.maximal_criticality import MaxCriticalityPolicy
from domoku.policies.radial import radial_2xnxn
from domoku.tools import GomokuTools as gt


class MaxInfluencePolicyParams(BaseModel):
    n: int  # board size
    radial_constr: list[float]
    radial_obstr: list[float]
    sigma: float  # Preference for offensive play, 1 >= lambda > 0
    iota: float  # The greed. Higher values make exploitation less likely. 50 is a good start


class MaxInfluencePolicy(tf.keras.Model, AbstractGanglion):
    """
    A policy that vaguely *feels* where the action is. This may help create reasonable
    trajectories in Deep RL approaches. The underlying CNN *measures* the radial influence
    of each stone on the board and counts the opponent's stones as obstructive.

    To be enable this policy to fight, you can supply a Criticality Model to override the soft advice produced here.
    """

    def winner(self, sample) -> Optional[int]:
        """
        :param sample: a board state sample
        :return: the winning channel if there is one else None
        """
        return self.crit_model.winner(sample)


    def eval(self, state: np.array):
        """
        :param state: nxnx4 np array containing the stones
        :return: Naive value: 1 if won, -1 if lost, 0 otherwise
        """
        winning_channel = self.crit_model.winner(state)
        if winning_channel is None:
            return 0

        if np.sum(state, axis=None) % 2 == self.pov:
            return -1.
        else:
            return 1.


    def __init__(self, params: MaxInfluencePolicyParams, pov: int,  # point of view - for value function
                 criticality_model: MaxCriticalityPolicy = None):
        super().__init__()
        self.params = params
        self.pov = pov
        self.n = params.n
        self.kernel_size = 2 * len(self.params.radial_constr) + 1
        self.filters = None
        self.biases = None
        self.occupied_suppression = -10.
        self.crit_model = criticality_model

        pot, agg = self.create_model()
        self.potential = pot
        self.aggregate = agg


    def create_model(self):
        self.construct_filters()
        len_radial = len(self.params.radial_constr)

        # Compute the current player's total potential, can be arbitrarily repeated
        # to create some forward-looking capabilities
        potential = tf.keras.layers.Conv2D(
            filters=len_radial, kernel_size=self.kernel_size,
            kernel_initializer=tf.constant_initializer(self.filters),
            bias_initializer=tf.constant_initializer(self.biases),
            activation=tf.nn.relu,
            padding='same',
            input_shape=(self.n, self.n, 4))

        aggregate = tf.keras.layers.Conv2D(
            filters=1, kernel_size=1,
            kernel_initializer=tf.constant_initializer([
                self.occupied_suppression, self.occupied_suppression, .6, .4]),
            bias_initializer=tf.constant_initializer(0.),
            activation=tf.nn.relu,
            padding='same',
            input_shape=(self.n, self.n, 4))

        return potential, aggregate


    def call(self, sample, move: Move = None):

        reshaped = np.reshape(sample, [-1, self.n, self.n, 4])
        soft = self.aggregate(
            self.potential(
                self.potential(
                    self.potential(reshaped))))

        if self.crit_model is not None:
            hard = self.crit_model.call(sample)
            res = hard * 10 + soft
        else:
            res = soft

        if move is not None:
            r, c = gt.b2m(move, self.n)
            return np.squeeze(res)[r][c]
        else:
            return res


    def sample(self, state, n=1):
        """
        Draw a sample from the distribution of moves provided by this policy.
        :param state: The NxNx4 board as provided by create_sample(board)
        :param n: The number of samples to return
        :return: A move to be played on that board (matrix - NOT board representation)
        """
        policy_result = self.call(state)
        n_choices = self.n * self.n
        choices = tf.nn.softmax(tf.reshape(policy_result, n_choices) * self.params.iota).numpy()
        elements = range(n_choices)
        probabilities = choices
        fields = np.random.choice(elements, n, p=probabilities)

        fields = [divmod(field, self.n) for field in fields]

        return fields


    def construct_filters(self):
        len_radial = len(self.params.radial_constr)
        kernel_size = 2 * len_radial + 1

        #  Determine the current player's offensive potential
        inf_stones_curr = radial_2xnxn(self.params.radial_constr, self.params.radial_obstr,
                                       self.occupied_suppression, self.occupied_suppression,
                                       gamma=1.0)
        inf_inf_curr = radial_2xnxn(self.params.radial_constr, self.params.radial_obstr,
                                    .9, .9,  # discounting for 2nd order influence
                                    gamma=.9)  # discounting for the time lag of the opponent - it's me first
        inf_curr = np.stack([inf_stones_curr, inf_inf_curr], axis=-2).reshape((kernel_size, kernel_size, 4))

        #  Determine the other player's offensive potential (for defense purposes)
        inf_stones_oth = radial_2xnxn(self.params.radial_obstr, self.params.radial_constr,
                                      self.occupied_suppression, self.occupied_suppression,
                                      gamma=1.0)
        inf_inf_oth = radial_2xnxn(self.params.radial_obstr, self.params.radial_constr,
                                   .9, .9,  # discounting for 2nd order influence
                                   gamma=.9)  # discounting for the time lag of the opponent - it's me first
        inf_oth = np.stack([inf_stones_oth, inf_inf_oth], axis=-2).reshape((kernel_size, kernel_size, 4))

        #  Projectors simply pass the stone channels through to the next layer
        zero = radial_2xnxn([0] * len_radial, None, 0, 0)
        proj_cur = radial_2xnxn([0] * len_radial, None, 1, 0)
        proj_cur = np.stack([proj_cur, zero], axis=-2).reshape((kernel_size, kernel_size, 4))

        proj_oth = radial_2xnxn([0] * 4, None, 0, 1)
        proj_oth = np.stack([proj_oth, zero], axis=-2).reshape((kernel_size, kernel_size, 4))

        filters = [proj_cur, proj_oth, inf_curr, inf_oth]
        self.biases = [0.] * len(filters)
        filters = np.stack(filters, axis=3)
        self.filters = np.reshape(filters, (self.kernel_size, self.kernel_size, 4, 4))
