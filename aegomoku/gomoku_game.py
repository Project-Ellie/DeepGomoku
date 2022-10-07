import abc
import copy
import random
from typing import Tuple, Optional

import numpy as np

from aegomoku.gomoku_board import GomokuBoard
from aegomoku.interfaces import Game, Move, TerminalDetector, SWAP2_FIRST_THREE, PASS, \
    SWAP2_AFTER_THREE, SWAP2_AFTER_FIVE, SWAP2_PASSED_THREE, SWAP2_PASSED_FIVE, FIRST_PLAYER, OTHER_PLAYER, GameState, \
    BLACK, SWAP2_DONE, Player
from aegomoku.policies.heuristic_policy import HeuristicPolicy
from aegomoku.policies.topological_value import TopologicalValuePolicy


class BoardInitializer:
    @abc.abstractmethod
    def initial_stones(self):
        pass


class TopoSwap2BoardInitializer(BoardInitializer):
    """
    TopoSwap2BoardInitializer produces boards of five stones with valuations close to
    half a sole stone's value, making it hard for the white player to choose between continuation and passing.
    """
    def __init__(self, board_size, num_searches=100, perimeter_width=5):
        """
        :param board_size: board_size
        :param num_searches: just a performance parameter, default is just fine
        :param perimeter_width: just a performance parameter, default is just fine
        """
        self.num_searches = num_searches
        self.board_size = board_size
        self.perimeter_width = perimeter_width

    def initial_stones(self):

        policy = TopologicalValuePolicy(kappa_s=6, kappa_d=5)
        board = GomokuBoard(self.board_size, stones=[200])
        d = self.perimeter_width

        # This is half the value of a board with a single black in the center
        center_value = policy.evaluate(board.canonical_representation())[1] / 2

        # Choose three stones somewhere in the perimeter area, so that the position is as neutral as can be
        corner = [i*self.board_size + j for i in range(d) for j in range(d)]
        left = [(i + d) * self.board_size + j for i in range(d) for j in range(d)]
        top = [i * self.board_size + j + d for i in range(d) for j in range(d)]
        perimeter = corner + left + top

        candidates = []
        dist = 1
        for i in range(self.num_searches):
            while True:
                moves = [board.Stone(random.choice(perimeter)) for _ in range(3)]
                try:
                    board = GomokuBoard(self.board_size, stones=moves)
                    value = policy.evaluate(board.canonical_representation())[1]
                    dist = (value - center_value) ** 2
                    break
                except AssertionError as e:
                    continue
            candidates.append((moves, dist))

        most_neutral = sorted(candidates, key=lambda e: e[1])[0][0]

        # Choose two more stones in the center area
        r = range(d, self.board_size-d+1)
        center = [i*self.board_size + j for i in r for j in r]

        candidates = []
        for i in range(self.num_searches):
            while True:
                moves = [board.Stone(random.choice(center)) for _ in range(2)]
                try:
                    moves = most_neutral + moves
                    board = GomokuBoard(self.board_size, stones=moves)
                    value = policy.evaluate(board.canonical_representation())[1]
                    dist = (value - center_value) ** 2
                    break
                except AssertionError as e:
                    continue
            candidates.append((moves, dist))

        as_list = sorted(candidates, key=lambda e: e[1])[0][0]
        as_string = ""
        for s in as_list:
            as_string += str(s)

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
    def __init__(self, board_size, initializer: BoardInitializer = None):
        super().__init__()
        self.board_size = board_size
        self.initializer = initializer
        self.n_in_row = 5
        self.detector: Optional[TerminalDetector] = None

    def get_initial_board(self) -> GomokuBoard:
        initial_stones = self.initializer.initial_stones() if self.initializer else ''
        return GomokuBoard(self.board_size, stones=initial_stones)

    def get_board_size(self, board) -> int:
        return self.board_size ** 2

    def get_action_size(self, board: GomokuBoard):
        # return number of actions
        return self.board_size ** 2

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
        state = board.canonical_representation()
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

        self.board = None
        self.phase = SWAP2_FIRST_THREE
        self.current_player = FIRST_PLAYER

    def __str__(self):
        player = "Player 1" if self.current_player == FIRST_PLAYER else "Player 2"
        color = "black" if self.board.get_current_color() == BLACK else "white"
        return f"{player} to move with {color}"

    __repr__ = __str__

    def get_current_player(self) -> int:
        """
        :return: the color of the original seating. Effectively, colors may change by passing
        """
        return self.current_player


    def get_next_player(self):
        if self.phase == SWAP2_FIRST_THREE:
            return FIRST_PLAYER

        elif self.phase == SWAP2_AFTER_THREE:
            return OTHER_PLAYER

        elif self.phase == SWAP2_PASSED_THREE:
            return FIRST_PLAYER

        elif self.phase == SWAP2_AFTER_FIVE:
            return FIRST_PLAYER

        elif self.phase == SWAP2_PASSED_FIVE:
            return OTHER_PLAYER

        else:
            self.current_player = 1 - self.current_player
            return self.current_player


    def transition(self, move: int) -> Tuple[int, int]:
        # update phase of the opening, return without change on the board if player passed
        # Swap the player - not the color

        if move == PASS:
            self.current_player = 1 - self.current_player
            if self.phase == SWAP2_AFTER_THREE:
                self.phase = SWAP2_PASSED_THREE
                return self.phase, self.current_player

            elif self.phase == SWAP2_AFTER_FIVE:
                self.phase = SWAP2_PASSED_FIVE
                return self.phase, self.current_player

        # same player continues
        if len(self.board.stones) in [1, 2, 4] and self.phase != SWAP2_PASSED_THREE:
            return self.phase, self.current_player

        elif len(self.board.stones) == 3:
            self.phase = SWAP2_AFTER_THREE

        elif len(self.board.stones) == 5:
            self.phase = SWAP2_AFTER_FIVE

        elif len(self.board.stones) == 6 or self.phase == SWAP2_PASSED_THREE and len(self.board.stones) == 4:
            self.phase = SWAP2_DONE

        self.current_player = self.get_next_player()

        return self.phase, self.current_player

    def get_phase(self):
        return self.phase


class Swap2(GomokuGame):
    """
    The board knows the phase, the game knows the rules
    """

    def __init__(self, board_size):
        super().__init__(board_size)

    @staticmethod
    def get_current_player(board) -> int:
        return board.get_current_player()


    def get_valid_moves(self, board: GomokuBoard):
        if len(board.stones) == 3 and board.get_phase() == SWAP2_AFTER_THREE:
            return [PASS] + super().get_valid_moves(board)
        elif len(board.stones) == 5 and board.get_phase() == SWAP2_AFTER_FIVE:
            return [PASS] + super().get_valid_moves(board)
        else:
            return super().get_valid_moves(board)


    def get_initial_board(self) -> GomokuBoard:
        initial_stones = self.initializer.initial_stones() if self.initializer else ''
        return GomokuBoard(self.board_size, game_state=Swap2GameState(), stones=initial_stones)


def one_game(game: GomokuGame, player1: Player, player2: Player, max_moves: int):
    """
    :param seqno: A sequence number for the game in the file
    :param game:
    :param player1: the player to make the first move
    :param player2: the other player
    :param eval_temperature: the temperature at which to read the MCTS scores
    :param max_moves: games are considered draw when no winner after this
    :return: tuple: Player1 name,
    """
    board = game.get_initial_board()
    player2.meet(player1)
    player = player1
    players = [player1, player2]
    num_stones = 0
    while game.get_winner(board) is None and num_stones < max_moves:

        board, move = player.move(board)
        next_player = board.get_current_player()
        player = players[next_player]

        print(f"{board}")
        if game.get_winner(board) is not None:
            break

    return player1.name, [s.i for s in board.get_stones()], game_data
