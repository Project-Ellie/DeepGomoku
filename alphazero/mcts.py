import copy
import logging
import math
from typing import Dict

import numpy as np

from alphazero.hr_tree import TreeNode
from alphazero.interfaces import Game, NeuralNet, Board

EPS = 1e-8

log = logging.getLogger(__name__)


class MCTS:
    """
    This class handles the MCTS tree.
    """

    def __init__(self, game: Game, nnet: NeuralNet, cpuct: float, num_simulations: int,
                 model_threshold=.2, verbose=0):
        self.game = game
        self.nnet = nnet
        self.cpuct = cpuct
        self.num_simulations = num_simulations
        self.model_threshold = model_threshold

        self.Q = {}  # stores Q values for s,a (as defined in the paper)
        self.Nsa = {}  # stores #times edge s,a was visited
        self.Ns = {}  # stores #times board s was visited
        self.Ps = {}  # stores initial policy (returned by neural net)

        self.Es = {}  # stores game.get_game_ended ended for board s
        self.Vs = {}  # stores game.get_valid_moves for board s
        if verbose > 0:
            self.verbosity = verbose
            print(f"verbosity: {self.verbosity}")

        board = game.get_initial_board()
        key = board.get_string_representation()
        self.tree_nodes: Dict[str, TreeNode] = {key: TreeNode(None, key, None, str(board), None)}

    def get_action_prob(self, board: Board, temperature=1):
        """
        This function performs numMCTSSims simulations of MCTS starting from
        canonical_board.

        Returns:
            probs: a policy vector where the probability of the ith action is
                   proportional to Nsa[(s,a)]**(1./temperature)
        """
        original_board = board
        for i in range(self.num_simulations):
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
        next_board, _ = self.game.get_next_state(board, a)
        v = self.search(next_board)

        new_value = self.update_node_stats(s, a, v)
        self.update_tree_view(s, a, new_value, next_board)

        return -v


    def update_tree_view(self, s, a, v, board):
        """
        :param s: the canonical string rep of the prev state
        :param a: the action from prev to curr state
        :param v: the value of the new state
        :param board: human-readable string rep of the current state
        """
        parent = self.tree_nodes[s]
        if parent.children.get(a) is not None:
            parent.children.get(a).v = v
        else:
            child = self.tree_nodes[s].add_child(a, board, v)
            self.tree_nodes[child.key] = child

    def update_node_stats(self, s, a, v):
        """
        :return: the updated average value for use in documentation and forensics
        """
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


    def probable_actions(self, board: Board):
        pi, _ = self.nnet.predict(board.canonical_representation())
        mx = np.max(pi)
        return list(np.where(pi >= self.model_threshold * mx)[0])


    def best_act(self, board: Board, s: str) -> int:
        # pick the action with the highest upper confidence bound from the probable actions
        # We're reducing the action space to those actions deemed probable by the model
        valids = self.Vs[s]
        cur_best = -float('inf')
        best_act = -1

        probable_actions = self.probable_actions(board)
        if len(probable_actions) == 1:
            return int(probable_actions[0])

        # for a in range(self.game.get_action_size(board)):
        for a in probable_actions:
            if valids[a]:
                if (s, a) in self.Q:
                    q = self.Q[(s, a)]
                    p = self.Ps[s][a]
                    sns = math.sqrt(self.Ns[s])
                    nsa = 1 + self.Nsa[(s, a)]
                    u = q + self.cpuct * p * sns / nsa
                else:
                    p = self.Ps[s][a]
                    sns = math.sqrt(self.Ns[s] + EPS)
                    u = self.cpuct * p * sns  # Q = 0 ?

                if u > cur_best:
                    cur_best = u
                    best_act = a

        return int(best_act)
