import numpy as np
from alphazero.interfaces import Game, TerminalDetector
from alphazero.gomoku_board import GomokuBoard


class GomokuGame(Game):
    def __init__(self, board_size, detector: TerminalDetector, initial: str = None):
        super().__init__()
        self.board_size = board_size
        self.board = None
        self.initial_stones = initial if initial is not None else ""
        self.n_in_row = 5
        self.detector = detector

    def get_initial_board(self):
        self.board = GomokuBoard(self.board_size, stones=self.initial_stones)
        return self.board.math_rep

    def get_board_size(self):
        return self.board_size ** 2

    def get_action_size(self):
        # return number of actions
        return self.board_size ** 2

    def get_next_state(self, action):
        self.board.act(action)
        return self.board.math_rep

    def get_valid_moves(self):
        return self.board.get_legal_actions()

    def get_game_ended(self):
        return self.detector.get_winner(self.board.canonical_representation())

    def get_canonical_form(self):
        return self.board.canonical_representation()

    # modified
    def get_symmetries(self, board, pi):

        # TODO: Implement this function according to spec

        # mirror, rotational
        assert(len(pi) == self.board_size**2 + 1)  # 1 for pass
        pi_board = np.reshape(pi[:-1], (self.board_size, self.board_size))
        symmetries = []

        for i in range(1, 5):
            for j in [True, False]:
                new_b = np.rot90(board, i)
                new_pi = np.rot90(pi_board, i)
                if j:
                    new_b = np.fliplr(new_b)
                    new_pi = np.fliplr(new_pi)
                symmetries += [(new_b, list(new_pi.ravel()) + [pi[-1]])]
        return symmetries

    def string_representation(self):
        # 8x8 numpy array (canonical board)
        return "".join([str(stone) for stone in self.board.get_stones()])

    def display(self):
        self.board.plot()
