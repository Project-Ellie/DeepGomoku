from typing import Callable, List
import tensorflow as tf
import numpy as np
from pydantic import BaseModel

from alphazero.policies.radial import radial_3xnxn, radial_2xnxn


class MaxInfluencePolicyParams(BaseModel):
    board_size: int  # board n
    radial_constr: List[float]
    radial_obstr: List[float]
    sigma: float  # Preference for offensive play, 1 >= sigma > 0
    iota: float  # The greed. Higher values make exploitation less likely. 50 is a good start


class NaiveInfluenceLayer(tf.keras.layers.Layer, Callable):

    def __init__(self, board_size):

        super().__init__()
        params = MaxInfluencePolicyParams(
            board_size=board_size,
            sigma=.7,
            iota=3,
            radial_constr=[.0625, .125, .25, .5],
            radial_obstr=[-.0625, -.125, -.25, -.5]
        )
        self.params = params
        self.input_size = board_size + 2
        self.kernel_size = 2 * len(self.params.radial_constr) + 1
        self.filters = None
        self.biases = None
        self.occupied_suppression = -100.

        self.potential = self.create_layers()


    def call(self, sample):
        # add two more channels filled with zeros. They'll be carrying the 'influence' of the surrounding stones.
        # That allows for arbitrarily deep chaining within our architecture
        n = self.input_size
        extended = np.concatenate([sample, np.zeros((n, n, 2))], axis=2).reshape((-1, n, n, 5))

        y = self.potential(extended)
        y = self.potential(y)
        y = self.potential(y)
        return y


    def create_layers(self):
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

        return potential


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
