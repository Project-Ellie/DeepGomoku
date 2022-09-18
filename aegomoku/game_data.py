import copy
from pickle import Unpickler
from typing import List, Tuple, Any, Callable

import numpy as np
import tensorflow as tf

from aegomoku.gomoku_board import GomokuBoard
from aegomoku.gomoku_game import GomokuGame
from aegomoku.interfaces import Game, Player, Board


def read_training_data(filename: str, condition: Callable = None):
    """
    Reads training data as
        stones: array(np.uint8) of flat board positions of subsequent moves up until that state
        probabilities: array of all probabilities on a uint8 scale 0-255
        values: float
    from file and returns eight (N+2)x(N+2) position symmetries and the corresponding probabilities and values
    as a straight array of (state, probabiltiy (float), value(float)) triples
    :param condition: a filter condition (s, p, v)->bool
    :param filename: file name
    :return: Array of (state, probabiltiy (float), value(float)) triples and an list of the game trajectories
    """
    all_examples = []
    all_games = []
    with open(filename, "rb") as file:
        while True:
            try:
                trajectory = Unpickler(file).load()
                all_examples += expand_trajectory(trajectory, condition)
                first, stones, _ = trajectory
                all_games.append((first, stones))

            except EOFError:
                return all_examples, all_games


def expand_trajectory(trajectory, condition: Callable = None):
    """
    read the training data from the trajectory record, ignore name and stones. They're for analysis purposes.
    :param trajectory: A record like (name, stones, positions) where positions are like (stones, probs, value)
    :param condition: a predicate function that defines whether to consider a particular tuple of (stones, prob, value)
    :return: fully expanded (from symmetries) examples (state, pred, value)
    """
    examples = []

    # We keep supporting trajectories without headers
    try:
        _, _, trajectory = trajectory
    except ValueError:
        trajectory = trajectory

    for position in trajectory:
        stones, probs, value = position
        if condition is not None:
            if not condition(stones, probs, value):
                continue
        board_size = np.sqrt(probs.shape[0]).astype(int)
        game = GomokuGame(board_size)
        probabilities = probs / 255.
        state = GomokuBoard(board_size, stones).canonical_representation()
        symmetries = game.get_symmetries(state, probabilities)
        for state, prediction in symmetries:
            examples.append((state, prediction, value))
    return examples


def create_dataset(data: List[Tuple[Any, Any, Any]], batch_size=1024, shuffle=True):
    subset = data
    x = np.asarray([t[0] for t in subset], dtype=float)
    pi = np.asarray([t[1] for t in subset])
    v = np.asarray([t[2] for t in subset])
    x_ds = tf.data.Dataset.from_tensor_slices(x).batch(batch_size)
    pi_ds = tf.data.Dataset.from_tensor_slices(pi).batch(batch_size)
    v_ds = tf.data.Dataset.from_tensor_slices(v).batch(batch_size)
    all_ds = tf.data.Dataset.zip((x_ds, pi_ds, v_ds))
    if shuffle:
        all_ds = all_ds.shuffle(buffer_size=batch_size)
    return all_ds


def one_game(game: Game, player1: Player, player2: Player,
             eval_temperature: float, max_moves: int, seqno: int = 0):
    """
    :param seqno: A sequence number for the game in the file
    :param game:
    :param player1: the player to make the first move
    :param player2: the other player
    :param eval_temperature: the temperature at which to read the MCTS scores
    :param max_moves: games are considered draw when no winner after this
    :return: tuple: Player1 name,
    """
    game_data = []
    board = game.get_initial_board()
    player2.meet(player1)
    player = player1
    players = [player1, player2]
    num_stones = 0
    while game.get_winner(board) is None and num_stones < max_moves:
        num_stones += 1
        prev_board = copy.deepcopy(board)
        board, move = player.move(board)

        print(f"{player.name}: {board}")
        if game.get_winner(prev_board) is not None:
            break

        example = create_example(prev_board, player, eval_temperature)
        game_data.append(example)

        player_index = board.get_current_player()
        player = players[player_index]

    return player1.name, [s.i for s in board.get_stones()], game_data


def create_example(the_board: Board, player: Player, temperature: float):
    """
    Create a single board image with the player's (MCTS-based) evaluation, ready for training
    """
    position = [stone.i for stone in the_board.get_stones()]
    # state = np.expand_dims(the_board.canonical_representation(), 0).astype(float)
    probs, value = player.evaluate(the_board, temperature)
    probs = (np.array(probs)*255).astype(np.uint8)
    return position, probs, value
