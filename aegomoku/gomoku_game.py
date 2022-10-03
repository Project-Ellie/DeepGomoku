import copy
import random
from typing import Tuple, Optional, List

import numpy as np
from keras import Model

from aegomoku.gomoku_board import GomokuBoard, expand
from aegomoku.interfaces import Game, Move, TerminalDetector, SWAP2_FIRST_THREE, \
    SWAP2_AFTER_THREE, SWAP2_AFTER_FIVE, FIRST_PLAYER, GameState, \
    BLACK, OpeningBook, BoardInitializer, GAMESTATE_NORMAL, SWAP2_AFTER_FOUR, Board, SWAP2_PASSED_THREE, \
    SWAP2_PASSED_FIVE
from aegomoku.policies.heuristic_policy import HeuristicPolicy


class Swap2RandomOpeningBook(OpeningBook):
    """
    Produces random moves during the opening phase.
    Intended for producing initial trajectories - not good for playing!
    """

    def __init__(self, board_size, num_probs: int = 20, pass_prob: float = 0.5):
        """
        :param board_size: the board size
        :param num_probs: number of
        :param pass_prob: probability of passing if allowed
        """
        self.board_size = board_size
        self.num_probs = num_probs
        self.action_size = self.board_size * self.board_size
        self.pass_prob = pass_prob
        self.Stone = GomokuBoard(self.board_size).Stone

    def next_move(self, state):
        field, phase = state
        if phase == GAMESTATE_NORMAL:
            return None
        return self.Stone(random.randint(0, self.action_size))

    def next_moves(self, state):
        field, phase = state
        if phase == GAMESTATE_NORMAL:
            return None
        return [self.Stone(random.randint(0, self.action_size - 1)) for _ in range(self.num_probs)]

    def set_policy(self, policy: Model):
        """
        Not required for this opening book. If you indeed meant to call this method,
        you may want to consider other opening books.
        :param policy: irrelevant
        """
        pass

    def should_pass(self, state):
        """
        Recommends pass or (after stone 4) yield on a random basis
        """
        field, phase = state
        if phase in [SWAP2_AFTER_THREE, SWAP2_AFTER_FOUR, SWAP2_AFTER_FIVE]:
            if random.random() < self.pass_prob:
                return True
            else:
                return False


class Swap2OpeningBook(OpeningBook):
    """
    Swap2OpeningBook produces move probabilities for Swap2 openings.
    half a sole stone's value, making it hard for the white player to choose between continuation and passing.
    This produces positions that are typically found after 5 moves in a swqp2 game.
    """

    def __init__(self, board_size, num_searches=100, perimeter_width=5,
                 policy: Model = None):
        """
        :param board_size: board_size
        :param num_searches: just a performance parameter, default is just fine
        :param perimeter_width: just a performance parameter, default is just fine
        :param policy: Adviser - may be injected later.
        """
        self.num_searches = num_searches
        self.board_size = board_size
        self.perimeter_width = perimeter_width
        self.game = Swap2(self.board_size)
        self.policy = policy  # must be injected later
        self.central_value = None  # must be injected later
        self.center = self.board_size * self.board_size // 2
        self.first_three = None
        self.two_more = None


    def should_pass(self, state):
        """
        :param state: NxNx3 state.
        :return: a recommendation to pass or not. Note that this is just
            a naive recommendation, based on the current value estimate.
        """
        _, phase = state
        if phase not in [SWAP2_AFTER_THREE, SWAP2_AFTER_FOUR, SWAP2_AFTER_FIVE]:
            return False
        _, value = self.policy.call(expand(state))
        if value < self.central_value:
            return True
        else:
            return False


    def set_policy(self, policy):
        self.policy = policy
        board = GomokuBoard(self.board_size, stones=[self.center])
        self.central_value = self.policy.advise(board.canonical_representation())[1] / 2


    def sample_neutral_3(self):
        d = self.perimeter_width
        # Choose three stones somewhere in the perimeter area, so that the position is as neutral as can be
        corner = [i*self.board_size + j for i in range(d) for j in range(d)]
        left = [(i + d) * self.board_size + j for i in range(d) for j in range(d)]
        top = [i * self.board_size + j + d for i in range(d) for j in range(d)]
        perimeter = corner + left + top

        board = GomokuBoard(self.board_size, stones=[self.center])

        candidates = []
        dist = 1
        for i in range(self.num_searches):
            while True:
                moves = [board.Stone(random.choice(perimeter)) for _ in range(3)]
                try:
                    board = GomokuBoard(self.board_size, stones=moves)
                    value = self.policy.advise(board.canonical_representation())[1]
                    dist = (value - self.central_value) ** 2
                    break
                except AssertionError:
                    continue
            candidates.append((moves, dist))

        return sorted(candidates, key=lambda e: e[1])[0][0]


    def sample_neutral_extra_2(self, most_neutral_3):
        d = self.perimeter_width
        board = GomokuBoard(self.board_size)

        # Choose two more stones in the center area
        r = range(d, self.board_size-d+1)
        center = [i*self.board_size + j for i in r for j in r]
        dist = 100.
        candidates = []
        for i in range(self.num_searches):
            while True:
                moves = [board.Stone(random.choice(center)) for _ in range(2)]
                try:
                    moves = most_neutral_3 + moves
                    board = GomokuBoard(self.board_size, stones=moves)
                    value = self.policy.advise(board.canonical_representation())[1]
                    dist = (value - self.central_value) ** 2
                    break
                except AssertionError:
                    continue
            candidates.append((moves, dist))

        return sorted(candidates, key=lambda e: e[1])[0][0]


    def next_moves(self, state):
        next_move = self.next_move(state)
        return [next_move] if next_move is not None else None

    def next_move(self, state):
        """
        Provide next move from the opening book. Here's the idea:
        Random sample a good combination of three, then pick from that - one by one.
        Do that for the next two stones if the player decides to add two more.
        :param state:
        :return: None if not in opening state
        """
        field, phase = state
        if phase == GAMESTATE_NORMAL:
            return None
        num_stones = np.sum(field[:, :, :2], axis=None)
        if num_stones == 0:
            if self.first_three is None:
                self.first_three = self.sample_neutral_3()
            return self.first_three[0]
        elif num_stones == 1:
            return self.first_three[1]
        elif num_stones == 2:
            return self.first_three[2]
        elif num_stones == 3:
            if self.two_more is None:
                self.two_more = self.sample_neutral_extra_2(self.first_three)[3:]
            return self.two_more[0]
        elif num_stones == 4:
            return self.two_more[1]
        else:
            return None


class Swap2BoardInitializer(BoardInitializer, Swap2OpeningBook):

    def __init__(self, policy, board_size, num_searches=100, perimeter_width=5):
        super().__init__(board_size, num_searches, perimeter_width, policy)


    def initial_stones(self):

        most_neutral_3 = self.sample_neutral_3()

        most_neutral_extra_2 = self.sample_neutral_extra_2(most_neutral_3)

        as_list = most_neutral_3 + most_neutral_extra_2

        as_string = ""
        for c in as_list:
            as_string += str(c)

        return as_string


class ConstantBoardInitializer(BoardInitializer):

    def __init__(self, stones: str):
        self.stones = stones

    def initial_stones(self):
        return self.stones


class RandomBoardInitializer(BoardInitializer):

    def __init__(self, board_size, num_stones, left=0, right=None, upper=0, lower=None):
        self.right = right if right is not None else board_size - 1
        self.lower = lower if lower is not None else board_size - 1
        self.left = left
        self.upper = upper
        self.num_stones = num_stones
        self.board_size = board_size

    def initial_stones(self):
        stones = ""
        n_it = 0
        for _ in range(self.num_stones):
            while True and n_it < self.board_size ** 2:
                col = random.randint(self.left, self.right)
                row = random.randint(self.upper, self.lower)
                stone = chr(col + 65) + str(self.board_size-row)
                if stone not in stones:
                    stones += stone
                    break
            if n_it == self.board_size ** 2:
                raise ValueError(f"Tried {self.board_size ** 2} times but failed.")
        return stones


class GomokuGame(Game):

    def __init__(self, board_size, n_special: int = 0, initializer: BoardInitializer = None):
        """
        :param board_size: length of a board side
        :param n_special: number of special moves not represented by positions
        :param initializer: a board initializer
        """
        super().__init__()
        self.board_size = board_size
        self.initializer = initializer
        self.n_in_row = 5
        self.detector: Optional[TerminalDetector] = None
        self.n_special = n_special


    def is_regular_move(self, board: Board, move: Move):
        return board.board_size * board.board_size >= move.i >= 0


    def get_initial_board(self) -> GomokuBoard:
        initial_stones = self.initializer.initial_stones() if self.initializer else ''
        return GomokuBoard(self.board_size, stones=initial_stones)


    def get_num_fields(self) -> int:
        return self.board_size * self.board_size


    def get_action_size(self, board: GomokuBoard):
        # return number of actions including special moves
        return self.get_num_fields() + self.n_special

    def get_next_state(self, board: GomokuBoard, action: Move) -> Tuple[GomokuBoard, int]:
        """
        computes the next state from a deep copy. Leaves the passed board unchanged
        :return:
        """
        board = copy.deepcopy(board)
        board.act(action)
        next_player = board.get_current_player()
        return board, next_player

    def get_valid_moves(self, board: GomokuBoard):
        bits = np.zeros([self.board_size * self.board_size])
        legal_indices = board.get_legal_actions()
        bits[legal_indices] = 1
        return bits

    def get_winner(self, board: GomokuBoard):
        if self.detector is None:
            self.detector = HeuristicPolicy(self.board_size)
        state, _ = board.canonical_representation()
        inputs = np.expand_dims(state, 0).astype(float)
        return self.detector.get_winner(inputs)

    # modified
    def get_symmetries(self, board, pi):
        """
        :param board: np array with the stones
        :param pi: np array containing the move probabilities for the current player
        :return:
        """
        # TODO: Implement this function according to spec

        # mirror, rotational
        n = self.board_size
        assert np.shape(pi) == (n * n,) or np.shape(pi) == (n, n), "pi should be square or flattened."
        if np.shape(pi) == (n * n,):
            pi_board = np.reshape(pi, (self.board_size, self.board_size))
        else:
            pi_board = pi
        symmetries = []

        for i in range(1, 5):
            for j in [True, False]:
                new_b = np.rot90(board, i, axes=(0, 1))
                new_pi = np.rot90(pi_board, i, axes=(0, 1))
                if j:
                    new_b = np.fliplr(new_b)
                    new_pi = np.fliplr(new_pi)
                symmetries += [(new_b, list(new_pi.ravel()))]
        return symmetries


class Swap2GameState(GameState):

    def __init__(self):
        self.phase = SWAP2_FIRST_THREE
        self.current_player = FIRST_PLAYER
        self.PASS = None  # we need the board_size to define that move's ordinal.F
        self.YIELD = None
        super().__init__()

    def link(self, board: Board):
        self.PASS = board.board_size * board.board_size
        self.YIELD = self.PASS + 1
        super().link(board)

    def __str__(self):
        player = "Player 1" if self.current_player == FIRST_PLAYER else "Player 2"
        color = "black" if self.board.get_current_color() == BLACK else "white"
        return f"{player} to move with {color}"

    def get_special_moves(self):
        return [self.PASS, self.YIELD]

    __repr__ = __str__

    def get_current_player(self) -> int:
        """
        :return: 0 for the player originally starting with black, 1 for the other
        """
        return self.current_player


    def transition(self, move: int) -> Tuple[List[int], int]:
        """
        Transition to the next state: Call this method *AFTER* the move!!
        :param move: the move that has just been made.
        :return: resulting phase and player *AFTER* the move.
        """
        # PASSED_... are an ephemeral states only needed for backpropagation
        if self.phase in [SWAP2_PASSED_THREE, SWAP2_PASSED_FIVE]:
            self.phase = GAMESTATE_NORMAL

        if self.phase in [GAMESTATE_NORMAL]:
            #
            # Normal game play - nothing special
            self.phase = GAMESTATE_NORMAL
            self.current_player = 1 - self.current_player
            return self.phase, self.current_player

        elif move == self.YIELD or isinstance(move, Move) and move.i == self.YIELD:
            """After setting stone number 4, white yields game play to black"""
            if self.phase == SWAP2_AFTER_FOUR:
                self.phase = GAMESTATE_NORMAL
                self.current_player = 1 - self.current_player
                return self.phase, self.current_player

        elif move == self.PASS or isinstance(move, Move) and move.i == self.PASS:
            #
            # Pass: Update phase of the opening
            # Swap the player - not the color
            self.current_player = 1 - self.current_player

            if self.phase == SWAP2_AFTER_THREE:
                self.phase = SWAP2_PASSED_THREE
                return self.phase, self.current_player

            elif self.phase == SWAP2_AFTER_FIVE:
                self.phase = SWAP2_PASSED_FIVE
                return self.phase, self.current_player

        else:
            #
            # Stones
            if len(self.board.stones) in [1, 2]:
                # We're in the first two moves of three
                return self.phase, self.current_player

            elif len(self.board.stones) == 3:
                self.phase = SWAP2_AFTER_THREE
                self.current_player = 1 - self.current_player
                return self.phase, self.current_player

            elif len(self.board.stones) == 4 and self.phase == SWAP2_AFTER_THREE:
                # We're in the first move of the second two
                self.phase = SWAP2_AFTER_FOUR
                return self.phase, self.current_player

            elif len(self.board.stones) == 5:
                self.phase = SWAP2_AFTER_FIVE
                self.current_player = 1 - self.current_player
                return self.phase, self.current_player

            else:
                self.phase = GAMESTATE_NORMAL
                self.current_player = 1 - self.current_player
                return self.phase, self.current_player


    def get_phase(self):
        return self.phase


class Swap2(GomokuGame):
    """
    The board knows the phase, the game knows the rules
    """

    def __init__(self, board_size, initializer=None):
        self.initializer = initializer
        super().__init__(board_size)


    def is_regular_move(self, board: GomokuBoard, move: Move):
        if move in [board.game_state.PASS, board.game_state.YIELD]:
            # special moves are: pass, yield
            return False
        else:
            # ...and parts of multi-stone moves
            field, phase = board.canonical_representation()
            num_stones = np.sum(field[:, :, :2], axis=None)
            return num_stones not in (0, 1, 3)

    @staticmethod
    def get_current_player(board) -> int:
        return board.get_current_player()


    def get_valid_moves(self, board: GomokuBoard):

        if len(board.stones) == 3 and board.get_phase() == SWAP2_AFTER_THREE:
            # seocond player may pass immediately
            return list(super().get_valid_moves(board)) + [1, 0]

        if len(board.stones) == 4 and board.get_phase() == SWAP2_AFTER_FOUR:
            # second player may choose to play 1 stone, then pass
            return list(super().get_valid_moves(board)) + [0, 1]

        elif len(board.stones) == 5 and board.get_phase() == SWAP2_AFTER_FIVE:
            # first player may pass after 2 moves from second
            return list(super().get_valid_moves(board)) + [1, 0]

        else:
            return list(super().get_valid_moves(board)) + [0, 0]


    def get_initial_board(self) -> GomokuBoard:
        initial_stones = self.initializer.initial_stones() if self.initializer else ''
        return GomokuBoard(self.board_size, game_state=Swap2GameState(), stones=initial_stones)

    def back_propagate(self, state, value):

        field, phase = state
        # When the player doesn't change, the value doesn't flip
        if phase in (SWAP2_FIRST_THREE, SWAP2_AFTER_FOUR):
            return value
        else:
            # that's the default minus sign
            return super().back_propagate(state, value)
