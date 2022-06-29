from typing import Optional, Callable

import numpy as np
from pydantic import BaseModel
import tensorflow as tf

from alphazero.interfaces import NeuralNet, LeadModel
from domoku.interfaces import AbstractGanglion
from domoku.policies.maximal_criticality import MaxCriticalityPolicy
from domoku.policies.radial import radial_3xnxn, radial_2xnxn
from domoku.policies.threat_search import ThreatSearchPolicy


class MaxInfluencePolicyParams(BaseModel):
    board_size: int  # board n
    radial_constr: list[float]
    radial_obstr: list[float]
    sigma: float  # Preference for offensive play, 1 >= sigma > 0
    iota: float  # The greed. Higher values make exploitation less likely. 50 is a good start


class MaxInfluencePolicy(tf.keras.Model, AbstractGanglion, LeadModel):
    """
    A policy that vaguely *feels* where the action is. This may help create reasonable
    trajectories in Deep RL approaches. The underlying CNN *measures* the radial influence
    of each stone on the board and counts the opponent's stones as obstructive.

    To be enable this policy to fight, you can supply a Criticality Model to override the soft advice produced here.
    """
    #
    #   TODO: Implement!
    #    Consider a policy with a wider exploration margin (possibly temperatur-based) and a cut-off to
    #    rule out sure-loss or plain-right ridiculous moves.
    #
    def get_reasonable_actions(self, state):
        """
        Returns a distribution of the most reasonable moves
        :return:
        """


    def __init__(self, params: MaxInfluencePolicyParams, pov: int,  # point of view - for value function
                 criticality_model: MaxCriticalityPolicy = None,
                 threat_model: ThreatSearchPolicy = None):
        super().__init__()
        self.params = params
        self.pov = pov
        self.input_size = params.board_size + 2
        self.kernel_size = 2 * len(self.params.radial_constr) + 1
        self.filters = None
        self.biases = None
        self.occupied_suppression = -10.
        self.crit_model = criticality_model
        self.threat_model = threat_model

        pot, agg, peel = self.create_model()
        self.potential = pot
        self.aggregate = agg
        self.peel = peel


    def soft(self, sample):
        # add two more channels filled with zeros. They'll be carrying the 'influence' of the surrounding stones.
        # That allows for arbitrarily deep chaining within our architecture
        n = self.input_size
        extended = np.concatenate([sample, np.zeros((n, n, 2))], axis=2).reshape((-1, n, n, 5))

        y = self.potential(extended)
        y = self.potential(y)
        y = self.potential(y)
        soft = self.peel(self.aggregate(y))
        return soft

    def call(self, sample, verbose=0, stone: Callable = None):

        soft = self.soft(sample)
        hard = self.crit_model.call(sample) if self.crit_model is not None else 0.
        threats = self.threat_model.call(sample) if self.threat_model is not None else 0.

        w_hard = 0.5
        w_threat = 1.0
        w_infl = .1

        res = w_hard * hard + w_threat * threats + w_infl * soft

        if verbose > 0:
            best = stone(int(np.argmax(hard))) if stone else np.argmax(hard)
            print(f"Critical:  {best} - {np.max(hard, axis=None):.5}, w={w_hard}")
            best = stone(int(np.argmax(threats))) if stone else np.argmax(threats)
            print(f"Threats:   {best} - {np.max(threats, axis=None):.5}, w={w_threat}")
            best = stone(int(np.argmax(soft))) if stone else np.argmax(soft)
            print(f"Influence: {best} - {np.max(soft, axis=None):.5}, w={w_infl}")
            best = stone(int(np.argmax(res))) if stone else np.argmax(res)
            print(f"Total:     {best} - {np.max(res, axis=None):.5}")

        return res


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


    def create_model(self):
        self.construct_filters()

        # Compute the current player's total potential, can be arbitrarily repeated
        # to create some forward-looking capabilities
        potential = tf.keras.layers.Conv2D(
            filters=5, kernel_size=self.kernel_size,
            kernel_initializer=tf.constant_initializer(self.filters),
            bias_initializer=tf.constant_initializer(self.biases),
            activation=tf.nn.relu,
            padding='same',
            input_shape=(self.input_size, self.input_size, 5))

        sigma = self.params.sigma  # offensiveness
        aggregate = tf.keras.layers.Conv2D(
            filters=1, kernel_size=1,
            kernel_initializer=tf.constant_initializer([
                self.occupied_suppression, self.occupied_suppression, self.occupied_suppression, sigma, 1-sigma]),
            bias_initializer=tf.constant_initializer(0.),
            activation=tf.nn.relu,
            padding='same',
            input_shape=(self.input_size-1, self.input_size-1, 5))

        peel = tf.keras.layers.Conv2D(
            filters=1, kernel_size=(3, 3),
            kernel_initializer=tf.constant_initializer([[0., 0., 0.], [0., 1., 0.], [0., 0., 0.]]),
            bias_initializer=tf.constant_initializer(0.),
            trainable=False)

        return potential, aggregate, peel


    def sample(self, state, n_sample=1, stone: Callable = None):
        """
        Draw a sample from the distribution of moves provided by this policy.
        :param stone: constructor, if you want a list of stones, rather than integers
        :param state: The NxNx3 math_rep of the board
        :param n_sample: The number of samples to return
        :return: A move to be played on that board (matrix - NOT board representation)
        """

        policy_result = self.call(state)
        n = self.params.board_size
        n_choices = n * n
        probabilities = tf.nn.softmax(tf.reshape(policy_result, n_choices) * self.params.iota).numpy()
        elements = range(n_choices)
        fields = np.random.choice(elements, n_sample, p=probabilities)

        if stone:
            fields = [stone(int(field)) for field in fields]

        return fields


    def construct_filters(self):
        """
        Five filters with five channels each. Like a 5x5 dense over each position.
        Two filters compute the influence, the others just project the input forward.
        This is designed so to give each layer the information about where exactly the stones are.
        I believe that's ok for a early-disposed heuristic auxiliary
        :return:
        """
        len_radial = len(self.params.radial_constr)

        #  Determine the current player's offensive potential
        inf_stones_curr = radial_3xnxn(self.params.radial_constr, self.params.radial_obstr, self.params.radial_obstr,
                                       self.occupied_suppression, self.occupied_suppression, self.occupied_suppression,
                                       gamma=1.0)
        inf_inf_curr = radial_2xnxn(self.params.radial_constr, self.params.radial_obstr,
                                    .9, .9,      # discounting for 2nd order influence
                                    gamma=.9)    # discounting for the time lag of the opponent - it's me first
        inf_curr = np.concatenate([inf_stones_curr, inf_inf_curr], axis=2)

        #
        #  Determine the other player's offensive potential (to determine the need of defense)
        inf_stones_oth = radial_3xnxn(self.params.radial_obstr, self.params.radial_constr, self.params.radial_obstr,
                                      self.occupied_suppression, self.occupied_suppression, self.occupied_suppression,
                                      gamma=1.0)
        inf_inf_oth = radial_2xnxn(self.params.radial_obstr, self.params.radial_constr,
                                   .9, .9,      # discounting for 2nd order influence
                                   gamma=.9)    # discounting for the time lag of my next move - it's him/her first
        inf_oth = np.concatenate([inf_stones_oth, inf_inf_oth], axis=2)

        #
        #  Projectors simply pass the stone channels through to the next layer
        zero = radial_2xnxn([0] * len_radial, None, 0, 0)

        proj_cur = radial_3xnxn([0] * len_radial, None, None, 1, 0, 0)
        proj_cur = np.concatenate([proj_cur, zero], axis=2)

        proj_oth = radial_3xnxn([0] * 4, None, None, 0, 1, 0)
        proj_oth = np.concatenate([proj_oth, zero], axis=2)

        proj_bnd = radial_3xnxn([0] * 4, None, None, 0, 0, 1)
        proj_bnd = np.concatenate([proj_bnd, zero], axis=2)

        filters = [proj_cur, proj_oth, proj_bnd, inf_curr, inf_oth]
        self.biases = [0.] * len(filters)
        filters = np.stack(filters, axis=3)
        self.filters = np.reshape(filters, (self.kernel_size, self.kernel_size, 5, 5))


class NeuralNetAdapter(NeuralNet, LeadModel):

    #
    #   Find a reasonable implementation for reasonable actions...;-)
    #
    def get_reasonable_actions(self, state):
        probs, _ = self.predict(state)
        max_prob = np.max(probs, axis=None)
        return probs[[probs > max_prob * 0.8]]

    def __init__(self, policy: Callable, *args):
        super().__init__(*args)
        self.policy = policy


    def train(self, examples, params):
        raise NotImplementedError


    def predict(self, state):
        output = self.policy(state)  # noqa
        board_size = self.policy.params.board_size  # noqa
        output = np.reshape(output, [board_size * board_size])

        # This 'Rescaling' produces a somewhat 'reasonable' distribution
        eps = 1e-8
        mx = np.max(output, axis=None)
        mn = np.min(output, axis=None)
        rescaled = np.log((output - mn) / (mx - mn) * np.e + eps)
        probs = tf.nn.softmax(rescaled)

        logits = tf.nn.tanh(output / 100.)
        value = tf.reduce_max(logits)

        return np.squeeze(probs), float(np.squeeze(value))


    def save_checkpoint(self, folder, filename):
        raise NotImplementedError


    def load_checkpoint(self, folder, filename):
        raise NotImplementedError
