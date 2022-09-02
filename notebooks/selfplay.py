import argparse
import os
import time

import ray
import logging

import yaml
from yaml import Loader

# import sys
# sys.path.extend(['/Users/wgiersche/workspace/Project-Ellie/DeepGomoku'])

from aegomoku.gomoku_game import RandomBoardInitializer, GomokuGame
from aegomoku.interfaces import MctsParams, PolicySpec
from aegomoku.policies.ray_impl import HeuristicRayPolicy
from aegomoku.ray.generic import RayFilePickler, SimpleCountingDispatcher, TaskMonitor
from aegomoku.ray.policy import SelfPlayDelegator, create_pool, PolicyRef
from aegomoku.self_play import SelfPlay

parser = argparse.ArgumentParser(description="Create Selfplay data with a given parameter config")
parser.add_argument('--params', '-p', action='store', default="selfplay_params.yaml",
                    help="Name of the parameters yaml file.")
parser.add_argument('--verbose', '-v', action='store',
                    help="Verbosity", default="0")

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


def startup_actors(params):
    """
    Instantiates and runs the entire zoo of actors
    """
    filename = params['process']['output']
    board_size = params['board']['size']
    num_trajectories = params['process']['workers']['trajectories']

    # File Writer
    the_writer = RayFilePickler.remote(filename, 'wb+')

    # Selfplay Dispatcher/Counter
    the_counter = SimpleCountingDispatcher.remote(num_trajectories)

    player = params['players'][0]
    workers = params['process']['workers']
    n_sp = workers['selfplay']
    n_p = workers['policy']

    # Policy Dispatcher
    policy_dispatcher = create_pool(num_workers=n_p, policy=HeuristicRayPolicy(),
                                    board_size=board_size, advice_cutoff=player['mcts']['advice_cutoff'])

    # Board and Game specifications
    initialize = params['board']['initialize']
    if initialize['type'] == 'random':
        stones = initialize['stones']
        left = initialize['left']
        right = initialize['right']
        upper = initialize['upper']
        lower = initialize['lower']
        initializer = RandomBoardInitializer(board_size=board_size, num_stones=stones,
                                             left=left, right=right, upper=upper, lower=lower)
    else:
        logger.warning("Only supporting random initializer for now.")
        exit(-1)
    game = GomokuGame(board_size, initializer)  # noqa: DS does not understand exit()

    # Selfplay params and actors
    mcts_params = MctsParams(
        cpuct=player['mcts']['cpuct'],
        num_simulations=player['mcts']['num_simulations'], temperature=0.3)

    selfplay_workers = [SelfPlay.remote(mcts_params=mcts_params) for _ in range(n_sp)]
    for selfplay in selfplay_workers:
        ray.get(selfplay.init.remote(board_size, game, PolicySpec(pool_ref=PolicyRef(policy_dispatcher))))

    # The monitor helps to block and report progress
    the_monitor = TaskMonitor.remote()

    # Selfplay delegators for each worker
    selfplay_delegators = [SelfPlayDelegator.remote(0, the_writer, the_counter, selfplay, the_monitor)
                           for selfplay in selfplay_workers]

    # Finally, start the delegators. They
    # - pick up a number from the counter
    # - make a blocking call the delegate - the selfplay worker, which
    #   - calls the policy dispatcher to evaluate policies on board positions, which
    #     - distributes the work evenly on it's fixed-size policy worker pool
    # - hand the result to the file writer
    # - report success to the monitor
    for worker in selfplay_delegators:
        worker.work.remote()

    # Block and report progress until all jobs are done
    progress = 0
    while progress < num_trajectories:
        new_progress = ray.get(the_monitor.get_status.remote())
        if new_progress > progress:
            print(new_progress)
            progress = new_progress
        else:
            print(".", end="")
            time.sleep(2.0)

    time.sleep(1)
    the_writer.close.remote()


def main():

    logger.info("Starting ray cluster")
    ray.init()

    logger.info(f"Reading params from file '{args.params}'")
    params = read_params(args.params)
    filename = params['process']['output']
    logger.info(f"Writing results to file: {filename}")

    startup_actors(params)

    logger.info("Shutting down ray cluster.")
    ray.shutdown()

    print("Done")


if __name__ == "__main__":
    main()
