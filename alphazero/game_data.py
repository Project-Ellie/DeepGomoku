from pickle import Unpickler

from alphazero.gomoku_board import GomokuBoard
from alphazero.gomoku_game import GomokuGame


def read_training_data(filename: str, board_size: int):
    """
    Reads training data as
        stones: array(np.uint8) of flat board positions of subsequent moves up until that state
        probabilities: array of all probabilities on a uint8 scale 0-255
        values: float
    from file and returns eight (N+2)x(N+2) position symmetries and the corresponding probabilities and values
    as a straight array of (state, probabiltiy (float), value(float)) triples
    :param filename: file name
    :param board_size: board size
    :return: Array of (state, probabiltiy (float), value(float)) triples
    """
    examples = []
    game = GomokuGame(board_size, None)
    with open(filename, "rb") as file:
        game_data = Unpickler(file).load()
        for traj in game_data:
            for position in traj:
                stones, probs, value = position
                probabilities = probs / 255.
                state = GomokuBoard(board_size, stones).canonical_representation()
                symmetries = game.get_symmetries(state, probabilities)
                for state, prediction in symmetries:
                    examples.append((state, prediction, value))
    return examples
