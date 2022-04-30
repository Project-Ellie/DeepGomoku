import numpy as np
import tensorflow as tf


class InfluenceReward(tf.keras.Model):
    """
    This function rewards an action a on state s with a scalar representing the gained total influence
    of the own stones vs the oppenent's stones' influence by that move
    """

    def __init__(self, board_size, current_influence, other_influence=None, tau=2.0, **kwargs):
        """

        :param current_influence: The array of the current player's influence values from remote to close
        :param other_influence: The array of the other player's influence values from remote to close,
            defaults to current_influence
        :param board_size: The size of the original board
        :param tau: The geometric coefficient to compute combined influence, defaults to 2.0
        :param kwargs: anything that needs to go to keras.Model
        """
        other_influence = other_influence if other_influence is not None else current_influence

        super().__init__(**kwargs)
        self.n_filters = 4
        # The range of the influence
        self.range_i = 4
        # sufficient padding for the given range
        self.input_size = 2 * self.range_i + board_size
        # geometric coefficient
        self.tau = tau
        self.max_reward = 10

        assert len(current_influence) == len(other_influence) == self.range_i, \
            f"Influence rays must have length {self.range_i}"

        self.filters = self.create_influence_filters(current_influence, other_influence)
        print(self.filters.shape)
        kernel_init = tf.constant_initializer(self.filters)

        self.input_layer = tf.keras.layers.Conv2D(filters=4, kernel_size=(2 * self.range_i + 1, 2 * self.range_i + 1),
                                                  kernel_initializer=kernel_init,
                                                  activation=tf.nn.relu,
                                                  input_shape=(self.input_size, self.input_size, 2))

        comb_init = tf.constant_initializer(np.ones(self.n_filters))
        self.combiner = tf.keras.layers.Conv2D(filters=1, kernel_size=(1, 1),
                                               kernel_initializer=comb_init,
                                               activation=tf.nn.relu)

        self.terminated = self.create_termination_detector()

    def create_termination_detector(self):
        five = [0, 0, 1, 1, 1, 1, 1, 0, 0]
        empty = np.zeros([9, 9], dtype=float)
        horiz = empty.copy()
        horiz[4, :] = five
        vert = empty.copy()
        vert[:, 4] = five

        filters = np.array([
            np.array([np.diag(five), empty]),
            np.array([np.diag(five)[::-1], empty]),
            np.array([horiz, empty]),
            np.array([vert, empty]),

            # np.array([empty, np.diag(five)]),
            # np.array([empty, np.diag(five)[::-1]]),
            # np.array([empty, horiz]),
            # np.array([empty, vert])
        ])
        filters = np.rollaxis(np.rollaxis(filters, 1, 4), 0, 4)
        kernel_init = tf.constant_initializer(filters)
        bias_init = tf.constant_initializer(-4.)

        return tf.keras.layers.Conv2D(filters=4, kernel_size=(9, 9),
                                      kernel_initializer=kernel_init, bias_initializer=bias_init,
                                      activation=tf.nn.relu, input_shape=(self.input_size, self.input_size, 2))


    def reshape(self, s, a):
        s = np.reshape(s, [-1, self.input_size, self.input_size, 2])
        a = np.reshape(a, [-1, self.input_size, self.input_size, 2])
        return s, a


    def call(self, s, a):
        s, a = self.reshape(s, a)
        current_player_after = self.input_layer(s + a)
        current_player_before = self.input_layer(s)
        combined = self.combine(current_player_after) - self.combine(current_player_before)

        if self.is_terminated(s):
            return -self.max_reward
        if self.is_terminated(s + a):
            return self.max_reward
        else:
            return tf.squeeze(tf.reduce_sum(combined)).numpy()

    def is_terminated(self, state):
        s = np.reshape(state, [-1, self.input_size, self.input_size, 2])
        return tf.reduce_sum(self.terminated(s)) >= 1

    def combine(self, influences):
        return self.combiner(influences ** self.tau)  # ** (1/self.tau)


    def influence_of(self, s, a):
        """
        Utility method to display the direct influence from the current player's point of view
        """
        s, a = self.reshape(s, a)
        current_player_after = self.input_layer(s + a)
        combined = self.combine(current_player_after)
        orig = tf.squeeze(combined)
        i_max = tf.reduce_max(orig)
        return tf.floor(orig / i_max * 9.99)


    def create_influence_filters(self, current_influence, other_influence):
        """
        Create filters for each general direction
        :param current_influence: array representing the influence of current player in any direction,
            like e.g. [1, 2, 4, 8]
        :param other_influence: array representing the influence of the other player in any direction,
            like e.g. [1, 2, 4, 8]
        :return: an np array representing that influence in a CNN
        """
        kernel_size = 2 * self.range_i + 1

        # Step 1: combine opposite directions and suppress influence on occupied fields
        supp_c = -sum(current_influence) * 8
        current = current_influence + [supp_c] + current_influence[::-1]

        supp_o = sum(other_influence) * 8
        other = other_influence + [supp_o] + other_influence[::-1]
        other = - np.array(other)

        ne = np.array([np.diag(current)[::-1], np.diag(other)[::-1]])
        nw = np.array([np.diag(current), np.diag(other)])
        w = np.zeros([2, kernel_size, kernel_size])
        w[0, self.range_i, :] = current
        w[1, self.range_i, :] = other
        n = np.zeros([2, kernel_size, kernel_size])
        n[0, :, self.range_i] = current
        n[1, :, self.range_i] = other
        return np.rollaxis(np.rollaxis(np.array([n, w, ne, nw]), 1, 4), 0, 4).astype(float)
