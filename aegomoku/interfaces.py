from __future__ import annotations

import abc
from typing import Tuple, Optional
from pydantic import BaseModel
import numpy as np


class MctsParams:

    def __init__(self, cpuct: float, num_simulations: int, advice_cutoff: float):
        self.cpuct = cpuct
        self.num_simulations = num_simulations
        self.advice_cutoff = advice_cutoff


class PolicySpec:
    """
    A means of telling remote actors how to instantiate their policy models.
    This is to avoid transfering the policy as an argument to a potentially
    remote actor, which will almost inevitably lead to serialization problems.
    """
    HEURISTIC = 'HEURISTIC'
    POOL_REF = 'POOL_REF'

    def __init__(self, checkpoint=None, pool_ref=None):
        """
        If no checkpoint is provided, heuristic policy is indicated
        :param checkpoint:
        """
        if checkpoint is not None:
            raise NotImplementedError("Can only support HEURISTIC for now.")
        elif pool_ref is not None:
            self.pool_ref = pool_ref
            self.type = PolicySpec.POOL_REF
        else:
            self.type = PolicySpec.HEURISTIC


class IllegalMoveException(Exception):
    def __int__(self, message):
        super().__init__(message)


class TrainParams(BaseModel):
    update_threshold: float  # During arena play, new neural net will be accepted if threshold or more games are won.
    max_queue_length: int    # Number of game examples to train the neural networks.

    epochs_per_train: int
    num_iterations: int
    num_episodes: int
    num_simulations: int     # Number of games moves for MCTS to simulate.
    arena_compare: int       # Number of games to play during arena play to determine if new net will be accepted.
    temperature_threshold: float
    cpuct: float
    checkpoint_dir: str
    load_model: bool
    load_folder_file: Tuple[str, str]
    num_iters_for_train_examples_history: int


class LeadModel:  # (abc.ABC):

    @abc.abstractmethod
    def get_advisable_actions(self, state):
        pass


class Move:
    i: int
    x: int
    y: int
    r: int
    c: int


class Board:  # (abc.ABC):

    @abc.abstractmethod
    def stone(self, pos: int) -> Move:
        pass

    @abc.abstractmethod
    def get_stones(self):
        pass

    @abc.abstractmethod
    def get_current_player(self):
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


class TerminalDetector(abc.ABC):

    @abc.abstractmethod
    def get_winner(self, board: Board):
        pass


class NeuralNet(LeadModel):  # , abc.ABC):
    """
    This class specifies the base NeuralNet class. To define your own neural
    network, subclass this class and implement the functions below. The neural
    network does not consider the current player, and instead only deals with
    the canonical form of the board.
    """

    def __init__(self, *args):
        pass

    @abc.abstractmethod
    def train(self, examples, params: TrainParams):
        """
        This function trains the neural network with examples obtained from
        self-play.

        Input:
            examples: a list of training examples, where each example is of form
                      (board, pi, v). pi is the MCTS informed policy vector for
                      the given board, and v is its value. The examples has
                      board in its canonical form.
        """
        pass

    @abc.abstractmethod
    def predict(self, board):
        """
        Input:
            board: current board in its canonical form.

        Returns:
            pi: a policy vector for the current board- a numpy array of length
                game.get_action_size
            v: a float in [-1,1] that gives the value of the current board
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
    def get_game_ended(self, board: Board):
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