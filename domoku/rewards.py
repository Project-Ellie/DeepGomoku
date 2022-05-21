import numpy as np
import tensorflow as tf

from domoku.data import create_binary_rep, create_binary_action


class InfluenceModel(tf.keras.Model):
    """
    This function rewards an action a on state s with a scalar representing the gained total influence
    of the own stones vs the oppenent's stones' influence by that move
    """


    def __init__(self, board_size, curr_tau, other_tau, current_influence, other_influence=None, **kwargs):
        """

        :param current_influence: The array of the current player's influence values from remote to close,
            first array of 4 is for vertical/horizontal, the second for diagonal directions
        :param other_influence: The array of the other player's influence values from remote to close,
            defaults to current_influence
        :param board_size: The size of the original board
        :param curr_tau: The geometric coefficient to compute combined influence, example: 2.0 for cartesion
        :param other_tau: The geometric coefficient to compute combined influence, example: 2.0 for cartesion
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
        self.curr_tau = curr_tau
        self.other_tau = other_tau
        self.max_reward = 100

        assert len(current_influence[0]) == len(other_influence[0]) == self.range_i, \
            f"Influence rays must have length {self.range_i}"

        self.filters = self.create_influence_filters(current_influence, other_influence)

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

        self.terminal_threat = self.create_lo4_detector()


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
        ])
        filters = np.rollaxis(np.rollaxis(filters, 1, 4), 0, 4)
        kernel_init = tf.constant_initializer(filters)
        bias_init = tf.constant_initializer(-4.)

        return tf.keras.layers.Conv2D(filters=4, kernel_size=(9, 9),
                                      kernel_initializer=kernel_init, bias_initializer=bias_init,
                                      activation=tf.nn.relu, input_shape=(self.input_size, self.input_size, 2))


    def create_lo4_detector(self):
        """
        Detects lines of 4 on a non-terminated board
        """
        five = [0, 0, 1, 1, 1, 1, 1, 0, 0]
        no_adv = -np.array(five)
        empty = np.zeros([9, 9], dtype=float)
        horiz = empty.copy()
        horiz[4, :] = five
        vert = empty.copy()
        vert[:, 4] = five

        filters = np.array([
            np.array([np.diag(five), np.diag(no_adv)]),
            np.array([np.diag(five)[::-1], np.diag(no_adv)[::-1]]),
            np.array([horiz, -horiz]),
            np.array([vert, -vert]),
        ])
        filters = np.rollaxis(np.rollaxis(filters, 1, 4), 0, 4)
        kernel_init = tf.constant_initializer(filters)
        bias_init = tf.constant_initializer(-3.)

        return tf.keras.layers.Conv2D(filters=4, kernel_size=(9, 9),
                                      kernel_initializer=kernel_init, bias_initializer=bias_init,
                                      activation=tf.nn.relu, input_shape=(self.input_size, self.input_size, 2))


    def reshape(self, s, a):
        s = np.reshape(s, [-1, self.input_size, self.input_size, 2])
        if a is not None:
            a = np.reshape(a, [-1, self.input_size, self.input_size, 2])
        return s, a


    def call(self, s, a, switch=False):
        s, a = self.reshape(s, a)
        current_player_after = self.input_layer(s + a)
        current_player_before = self.input_layer(s)
        tau = self.other_tau if switch else self.curr_tau
        combined = (self.combine(current_player_after, tau) -
                    self.combine(current_player_before, tau))

        if self.is_terminated(s):
            return -self.max_reward, True
        if self.is_terminated(s + a):
            return self.max_reward, True
        else:
            return tf.squeeze(tf.reduce_sum(combined)).numpy(), False


    def is_terminated(self, state):
        s = np.reshape(state, [-1, self.input_size, self.input_size, 2])
        return tf.reduce_sum(self.terminated(s)) >= 1


    def is_lo4_threat(self, state):
        s = np.reshape(state, [-1, self.input_size, self.input_size, 2])
        return tf.reduce_sum(self.terminal_threat(s)) >= 1


    def combine(self, influences, tau):
        return self.combiner(influences ** tau)  # ** (1/self.tau)


    def influence_of(self, state, for_humans=True):
        """
        Utility method to display the direct influence from the current player's point of view
        :param state: The two-layer binary representation of the board
        :param for_humans: if true, projects values to integers of the range [0, 9]
        """
        state, _ = self.reshape(state, None)
        current_player_after = self.input_layer(state)
        combined = self.combine(current_player_after, self.curr_tau)
        orig = tf.squeeze(combined)
        if for_humans:
            i_max = tf.reduce_max(orig)
            return tf.floor(orig / i_max * 9.99)
        else:
            return orig


    def create_influence_filters(self, current_influence, other_influence):
        """
        Create filters for each general direction
        :param current_influence: array representing the influence of current player in any direction,
            like e.g. [[1, 2, 4, 8], [1, 2, 4, 8]] for h/v and diag
        :param other_influence: array representing the influence of the other player in any direction,
            like e.g. [[1, 2, 4, 8], [1, 2, 4, 8]] for h/v and diag
        :return: an np array representing that influence in a CNN
        """
        kernel_size = 2 * self.range_i + 1

        # Step 1: combine opposite directions and suppress influence on occupied fields
        supp_c = -np.sum(np.array(current_influence)) * 8
        current_hv = current_influence[0] + [supp_c] + current_influence[0][::-1]
        current_diag = current_influence[1] + [supp_c] + current_influence[1][::-1]

        supp_o = np.sum(np.array(other_influence)) * 8
        other_hv = -(np.array(other_influence[0] + [supp_o] + other_influence[0][::-1]))
        other_diag = -(np.array(other_influence[1] + [supp_o] + other_influence[1][::-1]))

        ne = np.array([np.diag(current_diag)[::-1], np.diag(other_diag)[::-1]])
        nw = np.array([np.diag(current_diag), np.diag(other_diag)])
        w = np.zeros([2, kernel_size, kernel_size])
        w[0, self.range_i, :] = current_hv
        w[1, self.range_i, :] = other_hv
        n = np.zeros([2, kernel_size, kernel_size])
        n[0, :, self.range_i] = current_hv
        n[1, :, self.range_i] = other_hv
        return np.rollaxis(np.rollaxis(np.array([n, w, ne, nw]), 1, 4), 0, 4).astype(float)


class RewardContext:
    """
    Just a tool to examine the usefulness of reward models
    """


    def __init__(self, the_board, reward_model, offensiveness=0.5):
        self.board = the_board
        self.model = reward_model
        self.selector = InfluenceModel(board_size=self.board.N, curr_tau=2, other_tau=2,
                                       current_influence=[[1, 2, 4, 8], [1, 2, 4, 8]])
        self.offensiveness = offensiveness
        self.cut_off_offensive = 2
        self.cut_off_defensive = 2


    def reward(self, move):
        # current player's point of view
        state = create_binary_rep(self.board.N, self.board.stones, self.board.current_color,
                                  padding=4, border=True)
        action = create_binary_action(self.board.N, 4, move)
        offensive_reward, terminated = self.model(state, action)

        # adversary's point of view
        state_adv = create_binary_rep(self.board.N, self.board.stones, self.board.current_color,
                                      padding=4, border=True, switch=True)
        action_adv = create_binary_action(self.board.N, 4, move, switch=True)
        defensive_reward, terminated_adv = self.model(state_adv, action_adv)

        dangerous = self.model.is_lo4_threat(state_adv + action_adv)
        if dangerous and not terminated:
            return -self.model.max_reward, False

        o = self.offensiveness
        return 2 * o * offensive_reward - 2 * (1 - o) * defensive_reward, terminated or terminated_adv


    def find_candidates(self):
        n = self.board.N

        state = create_binary_rep(self.board, padding=4, border=True)
        me = self.selector.influence_of(state, None)
        offensive = {(me[i, j].numpy(), i, j) for i in range(n) for j in range(n)
                     if me[i, j] > self.cut_off_offensive}

        state_adv = create_binary_rep(self.board, padding=4, border=True, switch=True)
        other = self.selector.influence_of(state_adv, None)
        defensive = {(other[i, j].numpy(), i, j) for i in range(n) for j in range(n)
                     if other[i, j] > self.cut_off_defensive}

        return offensive.union(defensive)
