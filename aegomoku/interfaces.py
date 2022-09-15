from __future__ import annotations

import abc
from typing import Tuple, Optional
import numpy as np
from keras import models

GAMESTATE_NORMAL = 0

SWAP2_FIRST_THREE = -1
SWAP2_AFTER_THREE = -2
SWAP2_PASSED_THREE = -3
SWAP2_AFTER_FIVE = -4
SWAP2_PASSED_FIVE = -5
SWAP2_DONE = -6

PASS = -1


BLACK = 0
WHITE = 1

FIRST_PLAYER = 0
OTHER_PLAYER = 1


class GameState:
    @abc.abstractmethod
    def get_current_player(self):
        raise NotImplementedError("Please consider implementing this method.")

    def link(self, board: Board):
        self.board = board

    @abc.abstractmethod
    def get_phase(self):
        raise NotImplementedError("Please consider implementing this method.")

    @abc.abstractmethod
    def transition(self, move):
        raise NotImplementedError("Please consider implementing this method.")


class DefaultGomokuState(GameState):

    def __int__(self):
        self.board = None

    def get_phase(self):
        return GAMESTATE_NORMAL

    def link(self, board: Board):
        self.board = board

    def get_current_player(self):
        pass

    def transition(self, move):
        pass


class MctsParams:

    def __init__(self, cpuct: float, temperature: float, num_simulations: int):
        self.cpuct = cpuct
        self.num_simulations = num_simulations
        assert temperature == 0 or temperature > .1, "Temperatures near but not exactly zero are numerically instable."
        self.temperature = temperature


class PolicyParams:
    def __init__(self, model_file_name: Optional[str], advice_cutoff: float):
        self.model_file_name = model_file_name
        self.advice_cutoff = advice_cutoff


class IllegalMoveException(Exception):
    def __int__(self, message):
        super().__init__(message)


class Adviser:

    @abc.abstractmethod
    def get_advisable_actions(self, state):
        pass

    @abc.abstractmethod
    def evaluate(self, state):
        """
        :param state: current board in its canonical form NxNx3.

        :returns:
            pi: a policy vector for the current board- a numpy array of length
                game.get_action_size
            v: a float in [-1,1] that gives the value of the current board
        """
        pass


class PolicyAdviser(Adviser):

    def __init__(self, model: models.Model, params: PolicyParams):
        self.model = model
        self.params = params

    def get_advisable_actions(self, state):
        """
        :param state: nxnx3 representation of a go board
        :return:
        """
        probs, _ = self.model(state)
        max_prob = np.max(probs, axis=None)
        probs = np.squeeze(probs)
        advisable = np.where(probs > max_prob * self.params.advice_cutoff, probs, 0.)

        return [int(n) for n in advisable.nonzero()[0]]


    def evaluate(self, state):
        """
        :param state: nxnx3 representation of a go board
        :return:
        """
        inputs = np.expand_dims(state, 0).astype(float)
        p, v = self.model(inputs)
        return np.squeeze(p), np.squeeze(v)


class Move:
    r: int
    c: int
    field_size: int


    def __init__(self, x, y, i):
        self.x = x
        self.y = y
        self.i = i


class Board:

    def __int__(self, board_size: int):
        self.board_size = board_size

    def get_board_size(self):
        return self.board_size

    @abc.abstractmethod
    def stone(self, pos: int) -> Move:
        pass

    @abc.abstractmethod
    def get_stones(self):
        pass

    @abc.abstractmethod
    def get_string_representation(self):
        pass

    @abc.abstractmethod
    def get_legal_actions(self):
        """
        Returns all the legal moves for the current player
        """
        pass

    @abc.abstractmethod
    def get_legal_moves(self):
        pass

    @abc.abstractmethod
    def has_legal_moves(self):
        pass

    @abc.abstractmethod
    def act(self, *args):
        pass

    @abc.abstractmethod
    def canonical_representation(self):
        pass

    @abc.abstractmethod
    def plot(self):
        pass

    @abc.abstractmethod
    def get_current_color(self):
        pass

    @abc.abstractmethod
    def get_current_player(self):
        pass

    @abc.abstractmethod
    def get_phase(self):
        pass


class Player:  # (abc.ABC):

    opponent: Optional[Player]
    name: str

    def __eq__(self, other):
        if self.__class__ != other.__class__:
            return False
        return self.name == other.name

    def __hash__(self):
        return hash(self.name)

    @abc.abstractmethod
    def move(self, board: Board) -> Tuple[Board, Move]:
        """
        :param board: The board to put the next stone on
        :return: The same board instance with the additional Stone, and the Stone - for reference.
        """
        pass

    @abc.abstractmethod
    def meet(self, other: Player):
        pass

    @abc.abstractmethod
    def refresh(self):
        """
        Reset all persistent state to 'factory settings'
        :return: May return whatever is representing the new fresh state
        """
        pass

    @abc.abstractmethod
    def evaluate(self, board: Board, temperature: float):
        """
        Provide an opinion about the board - typically from MCTS stats
        :param board:
        :param temperature:
        :return:
        """
        pass


class TerminalDetector(abc.ABC):

    @abc.abstractmethod
    def get_winner(self, state):
        pass


class NeuralNet(Adviser):  # , abc.ABC):
    """
    This class specifies the base NeuralNet class. To define your own neural
    network, subclass this class and implement the functions below. The neural
    network does not consider the current player, and instead only deals with
    the canonical form of the board.
    """

    def __init__(self, *args):
        pass

    @abc.abstractmethod
    def train(self, train_examples, test_exaples=None, n_epochs=1):
        """
        This function trains the neural network with examples obtained from
        self-play.

        :param test_exaples:
        :param train_examples: a list of training examples, where each example is of form
                  (board, pi, v). pi is the MCTS informed policy vector for
                  the given board, and v is its value. The examples has
                  board in its canonical form.
        :param n_epochs
        """
        pass

    @abc.abstractmethod
    def save_checkpoint(self, folder, filename):
        """
        Saves the current neural network (with its parameters) in
        folder/filename
        """
        pass

    @abc.abstractmethod
    def load_checkpoint(self, folder, filename):
        """
        Loads parameters of the neural network from folder/filename
        """
        pass


class Game:  # (abc.ABC):
    """
    This class specifies the base Game class. To define your own game, subclass
    this class and implement the functions below. This works when the game is
    two-player, adversarial and turn-based.

    Use 1 for player1 and -1 for player2.

    See othello/OthelloGame.py for an example implementation.
    """
    board_size: int

    def __init__(self):
        pass


    @abc.abstractmethod
    def get_initial_board(self) -> Board:
        """
        Returns: a representation of the board (ideally this is the form
            that will be the input to your neural network). This may very well be a different board each time
        """
        pass

    @abc.abstractmethod
    def get_board_size(self, board: Board):
        """
        Returns:
            (x,y): a tuple of board dimensions
        """
        pass

    @abc.abstractmethod
    def get_action_size(self, board: Board):
        """
        Returns:
            actionSize: number of all possible actions
        """
        pass

    @abc.abstractmethod
    def get_next_state(self, board: Board, action: Move) -> Board:
        """
        Input:
            board: current board
            player: current player (1 or -1)
            move: move taken by current player

        Returns:
            nextBoard: board after applying move
            nextPlayer: player who plays in the next turn (should be -player)
        """
        pass

    @abc.abstractmethod
    def get_valid_moves(self, board: Board):
        """
        Input:
            board: current board

        Returns:
            validMoves: a binary vector of length self.get_action_size(), 1 for
                        moves that are valid from the current board and player,
                        0 for invalid moves
        """
        pass

    @abc.abstractmethod
    def get_winner(self, board: Board):
        """
        Input:
            board: current board
            player: current player (1 or -1)

        Returns:
            r: 0 if game has not ended. 1 if player won, -1 if player lost,
               small non-zero value for draw.
               
        """
        pass

    @abc.abstractmethod
    def get_symmetries(self, board: np.ndarray, pi):
        """
        Input:
            board: current board
            pi: policy vector of n self.get_action_size()

        Returns:
            symmForms: a list of [(board,pi)] where each tuple is a symmetrical
                       form of the board and the corresponding pi vector. This
                       is used when training the neural network from examples.
        """
        pass
