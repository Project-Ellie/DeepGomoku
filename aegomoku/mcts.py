import copy
import logging
import math
from typing import Dict, Tuple, List

import numpy as np

from aegomoku.hr_tree import TreeNode
from aegomoku.interfaces import Game, Adviser, Board, Move, MctsParams

EPS = 1e-8

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

        self.Q = {}  # stores Q values for s,a (as defined in the paper)
        self.Nsa = {}  # stores #times edge s,a was visited
        self.Ns = {}  # stores #times board s was visited
        self.Ps = {}  # stores initial policy (returned by neural net)

        self.Es = {}  # stores game.get_winner  for state s
        self.Vs = {}  # stores game.get_valid_moves for state s
        self.As = {}  # stores the policy's advice for state s
        if verbose > 0:
            self.verbosity = verbose
            print(f"verbosity: {self.verbosity}")

        board = game.get_initial_board()
        key = board.get_string_representation()
        self.root = TreeNode(None, key, None, board.get_stones(), None, 0, None)
        self.tree_nodes: Dict[str, TreeNode] = {
            key: self.root}

    def get_action_prob(self, board: Board, temperature=0.0):
        """
        This function performs numMCTSSims simulations of MCTS starting from
        canonical_board.
        """
        s = board.get_string_representation()
        advisable = self.As.get(s)
        if advisable is None:
            state = board.canonical_representation()
            advisable = self.adviser.get_advisable_actions(state)
            self.As[s] = advisable

        original_board = board
        for i in range(self.params.num_simulations):
            board = copy.deepcopy(original_board)
            self.search(board)

        return self.compute_probs(original_board, temperature)


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
        if sum(probs) == 0.0:
            print("Oops")
        return probs


    def search(self, board: Board, verbose=0):
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
        s = board.get_string_representation()

        # if we don't know whether this state is terminal...
        if s not in self.Es:
            # ... find out
            self.Es[s] = self.game.get_winner(board)

        # Now, if it is terminal, ...
        if self.Es[s] is not None:
            # Any defeat shows for the loser first - that just happened. Hence return -(-1)!
            return 1

        if s not in self.Ps:
            # we'll create a new leaf node and return the value estimated by the guiding player
            return -self.initialize_and_estimate_value(board, s)

        move, info = self.best_act(board=board, s=s)
        next_board, _ = self.game.get_next_state(board, move)
        v = self.search(next_board)

        # new_value = self.update_node_stats(s, move.i, v)
        self.update_node_stats(s, move.i, v)

        if self.game.is_regular_move(board, move):
            return -v
        else:
            return v


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


    def pot_best(self, board, s):
        return {board.stone(int(np.argmax(self.Ps[s], axis=None))): np.max(self.Ps[s], axis=None)}


    def initialize_and_estimate_value(self, board, s: str):
        """
        :param: board: the board
        :param: s, the canonical representation of the board (precomputed to save CPU cycles
        :return: The value of the leaf from the current player's point of view
        """

        # evaluate the policy for the move probablities and the value estimate.
        state = board.canonical_representation()
        self.Ps[s], v = self.adviser.advise(state)

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


    def probable_actions(self, board: Board) -> List:
        s = board.get_string_representation()
        advisable = self.As.get(s)
        if advisable is None or len(advisable) == 0:
            if s[-1] == 'P':
                pass
            inputs = board.canonical_representation()
            advisable = self.adviser.get_advisable_actions(inputs)
            if advisable is None or len(advisable) == 0:
                print("No advice?")
            effective_advisable = set(advisable).difference([s.i for s in board.get_stones()])
            if effective_advisable is None or len(effective_advisable) == 0:
                print("No advice?")

            self.As[s] = effective_advisable

        return [board.stone(i) for i in advisable]


    def best_act(self, board: Board, s: str) -> Tuple[Move, Dict]:
        # pick the move with the highest upper confidence bound from the probable actions
        # We're reducing the action space to those actions deemed probable by the model
        valids = self.Vs.get(s)
        if valids is None:
            print("oops.")
        cur_best = -float('inf')
        best_act = None

        probable_actions = self.probable_actions(board)
        if len(probable_actions) == 1:
            return probable_actions[0], {'u': float('inf'), 'q': None, 'p': 1, 'nsa': None}

        debug_info = {}
        for move in probable_actions:
            a = move.i  # use the integer representation
            if valids[a]:
                if (s, a) in self.Q:
                    u = self.ucb(s, a)
                else:
                    p = self.Ps[s][a]
                    sns = math.sqrt(self.Ns[s] + EPS)
                    u = self.params.cpuct * p * sns  # Q = 0 ?
                    debug_info[a] = {'u': u, 'q': None, 'p': p, 'nsa': None}

                if u >= cur_best:
                    cur_best = u
                    best_act = a

        if best_act is None:
            print("Oops!")

        return board.stone(best_act), debug_info


    def ucb(self, s: str, a: int):
        q = self.Q[(s, a)]
        p = self.Ps[s][a]
        sns = math.sqrt(self.Ns[s])
        nsa = 1 + self.Nsa[(s, a)]
        u = q + self.params.cpuct * p * sns / nsa
        return u
