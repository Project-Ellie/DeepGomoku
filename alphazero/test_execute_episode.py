import unittest
from alphazero.coach import Coach
from alphazero.gomoku_game import GomokuGame as Game
from alphazero.interfaces import TrainParams
from domoku.policies.maximal_criticality import MaxCriticalityPolicy
from domoku.policies.softadvice import MaxInfluencePolicy, MaxInfluencePolicyParams, NeuralNetAdapter
from domoku.constants import *

BOARD_SIZE = 15


class MyTestCase(unittest.TestCase):

    def test_something(self):

        params = TrainParams(
            update_threshold=0.6,
            max_queue_length=200000,    # Number of game examples to train the neural networks.
            num_simulations=25,
            arena_compare=40,         # Number of games to play during arena play to evaluate new network.
            cpuct=1.0,
            checkpoint_dir='./temperature/',
            load_model=False,
            load_folder_file=('/dev/models/8x100x50', 'best.pth.tar'),
            num_iters_for_train_examples_history=20,
            num_iterations=1000,
            num_episodes=100,
            temperature_threshold=15
        )

        detector = MaxCriticalityPolicy(BOARD_SIZE)

        game = Game(15, detector=detector, initial='H8')

        game.get_initial_board()

        brain = self.given_heuristic_brain()

        coach = Coach(game, brain, params)

        coach.execute_episode()

        self.assertTrue(False)

    @staticmethod
    def given_heuristic_brain():
        hard_policy = MaxCriticalityPolicy(board_size=BOARD_SIZE, overconfidence=5.0)
        params = MaxInfluencePolicyParams(
            board_size=BOARD_SIZE,
            sigma=.6,
            iota=6,
            radial_constr=[.0625, .125, .25, .5],
            radial_obstr=[-.0625, -.125, -.25, -.5]
        )
        policy = MaxInfluencePolicy(params, criticality_model=hard_policy, pov=BLACK)
        return NeuralNetAdapter(policy)
