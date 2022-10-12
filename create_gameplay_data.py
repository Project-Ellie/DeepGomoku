import argparse
import copy
import logging
import os
from pathlib import Path
from pickle import Pickler
import tensorflow as tf

import numpy as np
import yaml
from yaml import Loader

from aegomoku.gomoku_game import RandomBoardInitializer, GomokuGame, ConstantBoardInitializer
from aegomoku.gomoku_players import PolicyAdvisedGraphSearchPlayer
from aegomoku.interfaces import MctsParams, PolicyParams, Player, Board, Game, PolicyAdviser
from aegomoku.policies.topological_value import TopologicalValuePolicy

parser = argparse.ArgumentParser(description="Create Selfplay data with a given parameter config")
parser.add_argument('--params', '-p', action='store', default="gameplay_params.yaml",
                    help="Name of the parameters yaml file.")
parser.add_argument('--verbose', '-v', action='store',
                    help="Verbosity", default="0")
parser.add_argument('--seqno', '-s', action='store', type=int,
                    help="Sequence Number", default=0)
parser.add_argument('--info', '-i', action='store_true',
                    help="Dont actually act - just say", default=False)


args = parser.parse_args()

logger = logging.getLogger(__name__)

if args.verbose > "0":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s: %(message)s")


def read_params(filename):
    if not os.path.exists(filename):
        logger.warning(f"Parameter file: {filename} does not exist")
        exit(-1)
    with open(filename) as f:
        return yaml.load(f, Loader=Loader)


def create_game(params):
    board_size = params['board_size']
    opening = params['opening']
    initializer = opening.get('initializer')
    if initializer is not None:
        if initializer['type'] == 'random':
            stones = initializer['num_stones']
            left = initializer['left']
            right = initializer['right']
            upper = initializer['upper']
            lower = initializer['lower']
            initializer = RandomBoardInitializer(board_size=board_size, num_stones=stones,
                                                 left=left, right=right, upper=upper, lower=lower)
        elif initializer['type'] == 'constant':
            stones = initializer['stones']
            initializer = ConstantBoardInitializer(stones)

    return GomokuGame(board_size, initializer)


def prepare():
    pass


def create_gameplay_data(game, player1: Player, player2: Player, params, seqno):
    output_dir = Path.home() / params['output_dir'] / os.uname()[1]
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Writing results to directory: {output_dir}")
    temperature = params['eval_temperature']
    max_moves = params['max_moves']
    gameplay_data = []
    with open(os.path.join(output_dir, f"{seqno:05}.pickle"), 'wb+') as f:
        for seqno in range(params['num_trajectories_per_file']):
            one_game_data = one_game(seqno, game, player1, player2, temperature, max_moves)
            gameplay_data.append(one_game_data)
            Pickler(f).dump(one_game_data)

            player1, player2 = player2, player1
            player1.refresh()
            player2.refresh()

    return gameplay_data


def create_example(the_board: Board, player: Player, temperature: float):
    """
    Create a single board image with the player's (MCTS-based) evaluation, ready for training
    """
    position = [stone.i for stone in the_board.get_stones()]
    probs, value = player.evaluate(the_board, temperature)
    probs = (np.array(probs)*255).astype(np.uint8)
    return position, probs, value


def one_game(seqno: int, game: Game, player1: Player, player2: Player,
             eval_temperature: float, max_moves: int):
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
    num_stones = 0
    while game.get_winner(board) is None and num_stones < max_moves:
        num_stones += 1
        prev_board = copy.deepcopy(board)
        board, move = player.move(board)

        print(f"{seqno:02}: {board}")
        if game.get_winner(prev_board) is not None:
            break

        example = create_example(prev_board, player, eval_temperature)
        game_data.append(example)

        player = player.opponent

    return player1.name, [s.i for s in board.get_stones()], game_data


def create_players(game, player1, player2):

    players = []
    for player in [player1, player2]:
        name = player['name']
        cpuct = player['mcts']['cpuct']
        temperature = player['mcts']['temperature']
        num_simulations = player['mcts']['num_simulations']
        mcts_params = MctsParams(cpuct, temperature, num_simulations)
        advice = player['advice']

        if advice['type'] == 'TOPOLOGICAL_VALUE':
            advice_cutoff = advice['advice_cutoff']
            kappa_d = advice['kappa_d']
            kappa_s = advice['kappa_s']
            policy_params = PolicyParams(model_file_name=None, advice_cutoff=advice_cutoff)
            policy = TopologicalValuePolicy(kappa_s=kappa_s, kappa_d=kappa_d)
            adviser = PolicyAdviser(model=policy, params=policy_params)
        else:
            model_file = Path.home() / "workspace" / "Project-Ellie" / "DATA" / 'models' / advice['type']
            advice_cutoff = advice['advice_cutoff']
            policy_params = PolicyParams(model_file_name=model_file.as_posix(), advice_cutoff=advice_cutoff)
            model = tf.keras.models.load_model(model_file.as_posix())
            adviser = PolicyAdviser(model=model, params=policy_params)

        players.append(PolicyAdvisedGraphSearchPlayer(name, game, mcts_params, adviser=adviser))

    return players


def main():

    # np.warnings.filterwarnings('error', category=np.VisibleDeprecationWarning)
    seqno = args.seqno
    params = read_params(args.params)

    print()
    print("create_gameplay: ")
    print(f"Parameters from {os.path.join(os.getcwd(), args.params)}")
    print(f"Writing to: {params['process']['output_dir']}")

    if args.info:
        exit(0)

    game = create_game(params['game'])

    player1, player2 = create_players(game, *params['players'])

    create_gameplay_data(game, player1, player2, params['process'], seqno=seqno)

    logger.info("Shutting down .")

    print("Done")


if __name__ == "__main__":
    main()
