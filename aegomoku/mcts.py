import copy
import logging
import math
from typing import List

import numpy as np

from aegomoku.interfaces import Game, Adviser, Board, Move, MctsParams

log = logging.getLogger(__name__)


class MCTS:
    """
    This class handles the MCTS tree.
    """

    def __init__(self, game: Game, advisor: Adviser, params: MctsParams,
                 verbose=0):
        self.game = game
        self.adviser = advisor
        self.params = params

        self.Q = {}  # stores Q values for s,a
        self.Nsa = {}  # stores #times edge s,a was visited
        self.Ns = {}  # stores #times board s was visited
        self.Ps = {}  # stores initial policy (returned by neural net)

        self.Es = {}  # stores game.get_winner  for state s
        self.Vs = {}  # stores game.get_valid_moves for state s
        self.As = {}  # stores the policy's advice for state s
        if verbose > 0:
            self.verbosity = verbose
            print(f"verbosity: {self.verbosity}")


    def get_action_prob(self, board: Board, temperature=0.0):
        """
        Performs the simulations and returns the resulting probabilities as a 1-dim probability vector of length NxN
        """
        s = board.get_string_representation()
        advisable = self.As.get(s)
        if advisable is None:
            state = board.canonical_representation()
            advisable = self.adviser.get_advisable_actions(state)
            self.As[s] = advisable

        original_board = board
        print(f"Considering {self.params.num_simulations} positions:")
        for i in range(self.params.num_simulations):
            self.search(board)

        return self.compute_probs(original_board, temperature)


    def ponder(self, board: Board, num_simulations: int):
        for i in range(num_simulations):
            self.search(board)


    def compute_probs(self, board: Board, temperature: float):
        s = board.get_string_representation()
        counts = [self.Nsa[(s, a)] if (s, a) in self.Nsa else 0
                  for a in range(self.game.get_action_size(board))]

        if temperature == 0:
            best_as = np.array(np.argwhere(counts == np.max(counts))).flatten()
            best_a = np.random.choice(best_as)
            probs = [0] * len(counts)
            probs[best_a] = 1
            return probs

        counts = [x ** (1. / temperature) for x in counts]
        counts_sum = float(sum(counts)) or 1.0
        probs = [x / counts_sum for x in counts]
        return probs


    def search(self, board: Board, level=1):
        """
        This function performs one iteration of MCTS. It is recursively called
        till a leaf node is found. The move chosen at each node is one that
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

        board = copy.deepcopy(board)
        s = board.get_string_representation()

        # if I don't know whether this state is terminal...
        if s not in self.Es:
            # ... find out
            self.Es[s] = self.game.get_winner(board)

        # Now, if it is terminal, ...
        if self.Es[s] is not None:
            v = 1. if self.Es[s] == 0 else -1.
            return -v

        if s not in self.Ps:
            # we'll create a new leaf node and return the value estimated by the guiding player
            v = self.initialize_and_estimate_value(board, s)
            return -v

        advisable = self._advisable_actions(board)

        v0 = -1
        v = None
        while v0 == -1:
            move = self.best_act(board=board, s=s, choice=advisable)
            if move is None:
                print("Ain't got no move no mo'. Giving up.")
                return -1.0
            next_board, _ = self.game.get_next_state(board, move)

            # Go deeper
            v0 = self.search(next_board, level+1)
            if v0 == 1.:
                v = 1.
                self.Es[s] = 0
            elif v0 == -1.:
                v = -1.
            else:
                v = self.params.gamma * v0

            self.update_node_stats(s, move.i, v)

        if v == -1:
            self.Es[s] = 1

        return -v


    def update_node_stats(self, s, a, v):
        """
        :return: the updated average value for use in documentation and forensics
        """
        if v == 1:
            self.Ns[s] = 1e5  # Just an overwhelmingly big number
            self.Q[(s, a)] = 1
            self.Nsa[(s, a)] = self.Ns[s]
        else:
            if (s, a) in self.Q:
                self.Q[(s, a)] = (self.Nsa[(s, a)] * self.Q[(s, a)] + v) / (self.Nsa[(s, a)] + 1)
                self.Nsa[(s, a)] += 1
            else:
                self.Q[(s, a)] = v
                self.Nsa[(s, a)] = 1
            self.Ns[s] += 1

        return self.Q[(s, a)]


    def initialize_and_estimate_value(self, board, s: str):
        """
        :param: board: the board
        :param: s, the canonical representation of the board (precomputed to save CPU cycles
        :return: The value of the leaf from the current player's point of view
        """

        # evaluate the policy for the move probablities and the value estimate.
        inputs = board.canonical_representation()
        p, v = self.adviser.evaluate(inputs)
        self.Ps[s] = p

        advisable = self._advisable_actions(board)
        adv_mask = np.zeros(board.board_size * board.board_size)
        adv_mask[advisable] = 1.0

        # rule out illegal moves and renormalize
        valids = self.game.get_valid_moves(board)
        self.Ps[s] = self.Ps[s] * valids * adv_mask  # masking invalid moves
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


    def _advisable_actions(self, board: Board) -> List:
        s = board.get_string_representation()
        advisable = self.As.get(s)
        if advisable is None or len(advisable) == 0:
            inputs = board.canonical_representation()
            advisable = self.adviser.get_advisable_actions(inputs)
            advisable = set(advisable).difference([s.i for s in board.get_stones()])
            self.As[s] = advisable

        return list(advisable)

    def _advisable_stones(self, board: Board) -> List:
        advisable = self._advisable_actions(board)
        return [board.stone(i) for i in advisable]


    def best_act(self, board: Board, s: str, choice) -> Move:
        # pick the move with the highest upper confidence bound from the given choice
        # We're reducing the action space to those actions deemed probable by the model
        valids = self.Vs[s]
        cur_best = -float('inf')
        best_act = None

        if len(choice) == 1:
            return board.stone(choice[0])

        for a in choice:
            if valids[a]:
                if (s, a) in self.Q:
                    if self.Q[(s, a)] == -1.:
                        valids[a] = 0
                        continue
                    u, q, p, sns, nsa = self.ucb(s, a)
                else:
                    p = self.Ps[s][a]
                    # the 1e-10 is needed to distinguish by p when N is still zero.
                    sns = math.sqrt(self.Ns[s] + 1e-10)
                    u = self.params.cpuct * p * sns

                if u >= cur_best:
                    cur_best = u
                    best_act = a

        return board.stone(best_act)


    def ucb(self, s: str, a: int):
        q = self.Q[(s, a)]
        p = self.Ps[s][a]
        sns = math.sqrt(self.Ns[s] + 1e-10)
        nsa = 1 + self.Nsa[(s, a)]
        u = q + self.params.cpuct * p * sns / nsa
        return u, q, p, sns, nsa
