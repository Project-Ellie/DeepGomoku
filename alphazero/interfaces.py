import abc
from typing import Tuple
from pydantic import BaseModel


class IllegalMoveException(Exception):
    def __int__(self, message):
        super().__init__(message)


class TrainParams(BaseModel):
    update_threshold: float  # During arena play, new neural net will be accepted if threshold or more games are won.
    max_queue_length: int    # Number of game examples to train the neural networks.

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


class Board(abc.ABC):

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


class TerminalDetector(abc.ABC):

    @abc.abstractmethod
    def get_winner(self, board: Board):
        pass


class NeuralNet(abc.ABC):
    """
    This class specifies the base NeuralNet class. To define your own neural
    network, subclass this class and implement the functions below. The neural
    network does not consider the current player, and instead only deals with
    the canonical form of the board.

    See othello/brain.py for an example implementation.
    """

    def __init__(self, *args):
        pass

    @abc.abstractmethod
    def train(self, examples):
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


class Game(abc.ABC):
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
    def get_initial_board(self):
        """
        Returns:
            startBoard: a representation of the board (ideally this is the form
                        that will be the input to your neural network)
        """
        pass

    @abc.abstractmethod
    def get_board_size(self):
        """
        Returns:
            (x,y): a tuple of board dimensions
        """
        pass

    @abc.abstractmethod
    def get_action_size(self):
        """
        Returns:
            actionSize: number of all possible actions
        """
        pass

    @abc.abstractmethod
    def get_next_state(self, action):
        """
        Input:
            board: current board
            player: current player (1 or -1)
            action: action taken by current player

        Returns:
            nextBoard: board after applying action
            nextPlayer: player who plays in the next turn (should be -player)
        """
        pass

    @abc.abstractmethod
    def get_valid_moves(self):
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
    def get_game_ended(self):
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
    def get_canonical_form(self):
        """
        Input:
            board: current board
            player: current player (1 or -1)

        Returns:
            canonical_board: returns canonical form of board. The canonical form
                            should be independent of player. For e.g. in chess,
                            the canonical form can be chosen to be from the pov
                            of white. When the player is white, we can return
                            board as is. When the player is black, we can invert
                            the colors and return the board.
        """
        pass

    @abc.abstractmethod
    def get_symmetries(self, board, pi):
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

    @abc.abstractmethod
    def string_representation(self):
        """
        Input:
            board: current board

        Returns:
            boardString: a quick conversion of board to a string format.
                         Required by MCTS for hashing.
        """
        pass
