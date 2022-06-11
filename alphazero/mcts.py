import copy
import logging
import math

import numpy as np

from alphazero.interfaces import TrainParams, Game, NeuralNet, Board

EPS = 1e-8

log = logging.getLogger(__name__)


class MCTS:
    """
    This class handles the MCTS tree.
    """

    def __init__(self, game: Game, nnet: NeuralNet, args: TrainParams):
        self.game = game
        self.nnet = nnet
        self.args = args
        self.Q = {}  # stores Q values for s,a (as defined in the paper)
        self.Nsa = {}  # stores #times edge s,a was visited
        self.Ns = {}  # stores #times board s was visited
        self.Ps = {}  # stores initial policy (returned by neural net)

        self.Es = {}  # stores game.get_game_ended ended for board s
        self.Vs = {}  # stores game.get_valid_moves for board s

    def get_action_prob(self, board: Board, temperature=1):
        """
        This function performs numMCTSSims simulations of MCTS starting from
        canonical_board.

        Returns:
            probs: a policy vector where the probability of the ith action is
                   proportional to Nsa[(s,a)]**(1./temperature)
        """
        original_board = board
        for i in range(self.args.num_simulations):
            board = copy.deepcopy(original_board)
            self.search(board)

        s = original_board.get_string_representation()
        counts = [self.Nsa[(s, a)] if (s, a) in self.Nsa else 0
                  for a in range(self.game.get_action_size(original_board))]

        if temperature == 0:
            best_as = np.array(np.argwhere(counts == np.max(counts))).flatten()
            best_a = np.random.choice(best_as)
            probs = [0] * len(counts)
            probs[best_a] = 1
            return probs

        counts = [x ** (1. / temperature) for x in counts]
        counts_sum = float(sum(counts))
        probs = [x / counts_sum for x in counts]
        return probs

    def search(self, board: Board):
        """
        This function performs one iteration of MCTS. It is recursively called
        till a leaf node is found. The action chosen at each node is one that
        has the maximum upper confidence bound as in the paper.

        Once a leaf node is found, the neural network is called to return an
        initial policy P and a value v for the state. This value is propagated
        up the search path. In case the leaf node is a terminal state, the
        outcome is propagated up the search path. The values of Ns, Nsa, Q are
        updated.

        NOTE: the return values are the negative of the value of the current
        state. This is done since v is in [-1,1] and if v is the value of a
        state for the current player, then its value is -v for the other player.

        Returns:
            v: the negative of the value of the current canonical_board
        """

        s = board.get_string_representation()

        # if I don't know whether this state ist terminal...
        if s not in self.Es:
            # ... find out
            self.Es[s] = self.game.get_game_ended(board)

        # Now, if it is terminal, ...
        if self.Es[s] is not None:
            # ... return the negative value (which is the terminal state value, here)
            # Negative, because we tell the 'other' player the value from his/her point of view.
            return -self.Es[s]

        if s not in self.Ps:
            # we'll create a new leaf node and return the negative of the value estimated by the guiding player
            return -self.initialize_and_estimate_value(board, s)

        a = self.best_act(board=board, s=s)
        board_copy = copy.deepcopy(board)
        next_board, _ = self.game.get_next_state(board_copy, a)
        v = self.search(next_board)

        self.update_node_stats(s, a, v)

        return -v


    def update_node_stats(self, s, a, v):
        if (s, a) in self.Q:
            self.Q[(s, a)] = (self.Nsa[(s, a)] * self.Q[(s, a)] + v) / (self.Nsa[(s, a)] + 1)
            self.Nsa[(s, a)] += 1
        else:
            self.Q[(s, a)] = v
            self.Nsa[(s, a)] = 1
        self.Ns[s] += 1


    def initialize_and_estimate_value(self, board, s: str):
        """
        :param board: the board
        :param s, the canonical representation of the board (precomputed to save CPU cycles
        :return: The value of the leaf form the current player's point of view
        """

        # evaluate the policy for the action probablities and the value estimate.
        self.Ps[s], v = self.nnet.predict(board.canonical_representation())

        # rule out illegal moves and renormalize
        valids = self.game.get_valid_moves(board)
        self.Ps[s] = self.Ps[s] * valids  # masking invalid moves
        sum_ps_s = np.sum(self.Ps[s])
        if sum_ps_s > 0:
            self.Ps[s] /= sum_ps_s  # renormalize
        else:
            # if all valid moves were masked make all valid moves equally probable

            # NB! All valid moves may be masked if either your NNet architecture is insufficient
            # or you've get overfitting or something else.
            # If you have got dozens or hundreds of these messages
            # you should pay attention to your NNet and/or training process.
            log.error("All valid moves were masked, doing a workaround.")
            self.Ps[s] = self.Ps[s] + valids
            self.Ps[s] /= np.sum(self.Ps[s])

        self.Vs[s] = valids
        self.Ns[s] = 0
        return v


    def best_act(self, board: Board, s: str):
        # pick the action with the highest upper confidence bound
        valids = self.Vs[s]
        cur_best = -float('inf')
        best_act = -1

        #
        #  TODO: Here, we should cooperate with a lead policy that reduces the action space.
        #   In the spirit of mu-zero, that could be considered a part of the environment model.
        #
        for a in range(self.game.get_action_size(board)):
            if valids[a]:
                if (s, a) in self.Q:
                    u = self.Q[(s, a)] + self.args.cpuct * self.Ps[s][a] * math.sqrt(self.Ns[s]) / (
                            1 + self.Nsa[(s, a)])
                else:
                    u = self.args.cpuct * self.Ps[s][a] * math.sqrt(self.Ns[s] + EPS)  # Q = 0 ?

                if u > cur_best:
                    cur_best = u
                    best_act = a

        return best_act
