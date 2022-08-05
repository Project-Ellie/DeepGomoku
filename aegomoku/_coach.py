import logging
import os
import sys
from collections import deque
from pickle import Pickler, Unpickler
import random
import copy

import numpy as np
from tqdm import tqdm

from aegomoku.arena import Arena
from aegomoku.gomoku_model import NeuralNetAdapter
from aegomoku.interfaces import TrainParams, Board, Game
from aegomoku.mcts import MCTS

log = logging.getLogger(__name__)


class Coach:
    """
    This class executes the self-play + learning. It uses the functions defined
    in Game and NeuralNet. params are specified in main.py.
    """

    def __init__(self, game, params: TrainParams):
        self.game = game
        self.params = params
        self.iterations_queue = []  # history of examples from params.numItersForTrainExamplesHistory iterations
        self.skip_first_self_play = False  # can be overriden in load_train_examples()
        self.current_player = None
        self.checkpoint_prefix = 'checkpoint_'

    def execute_episode(self, expert: MCTS, with_moves=False):
        """
        Watching the expert (=current champion or so) play to learn the game.

        This function executes one episode of self-play, starting with player 1.
        As the game is played, each turn is added as a training example to
        train_examples. The game is played till the game ends. After the game
        ends, the outcome of the game is used to assign values to each example
        in train_examples.

        It uses a temperature=1 if episode_step < tempThreshold, and thereafter
        uses temperature=0.

        Returns:
            train_examples: a list of examples of the form (canonical_board, currPlayer, pi,v)
                           pi is the MCTS informed policy vector, v is +1 if
                           the player eventually won the game, else -1.
        """
        train_examples = []
        game = expert.game
        board = game.get_initial_board()
        current_player = board.get_current_player()
        episode_step = 0

        temperatures = [0, 1]  # more tight vs more explorative

        while True:
            episode_step += 1
            temperature = temperatures[episode_step % 2]

            pi = expert.get_action_prob(board, temperature=temperature)
            sym = expert.game.get_symmetries(board.canonical_representation(), pi)
            for b, p in sym:
                train_examples.append([b, current_player, p, None])

            action = np.random.choice(len(pi), p=pi)
            board, current_player = game.get_next_state(board, action)

            r = game.get_game_ended(board)

            if r is not None:
                train_examples = [(x[0], x[2], r * ((-1) ** (x[1] != current_player))) for x in train_examples]
                if with_moves:
                    return train_examples, self.recover_moves(train_examples)
                else:
                    return train_examples


    def recover_moves(self, examples):
        """
        Recover the sequence of moves from a trajectory's examples for debugging purposes
        :param examples:
        :return:
        """
        n_moves = len(examples)//8
        board = self.game.get_initial_board()
        moves = board.string_to_stones(self.game.initial_stones)
        example0 = board.math_rep
        for i in range(n_moves):
            example1 = examples[i * 8][0]
            example1_ = example1[:, :, [1, 0, 2]]
            move = list(np.argwhere(example1_ - example0 == 1)[0][:2] - [1, 1])
            example0 = example1
            moves.append(board.Stone(*move))
        return moves


    def create_trajectories(self, idol: MCTS, n_it: int, shuffle=True):
        """
        :param idol: the current champion
        :param n_it: the ordinal of the particular iteration
        :return: most recent training examples, containing some new and some old ones
            as a list of tuples (state, probs, value)
        """
        # examples of the iteration
        if not self.skip_first_self_play or n_it > 1:
            iteration = deque([], maxlen=self.params.max_queue_length)

            for _ in tqdm(range(self.params.num_episodes), desc="   Self Play"):
                #iteration += self.execute_episode(idol)
                iteration += self.observe_trajectory(idol)

            # save the iteration examples to the history
            self.iterations_queue.append(iteration)

        if len(self.iterations_queue) > self.params.num_iters_for_train_examples_history:
            log.warning(
                f"Removing the oldest entry in train_examples. len history = {len(self.iterations_queue)}")
            self.iterations_queue.pop(0)
        # backup history to a file
        # NB! the examples were collected using the model from the previous iteration, so (i-1)
        self.save_train_examples(n_it - 1)

        # shuffle examples before training
        train_examples = []
        for e in self.iterations_queue:
            train_examples.extend(e)

        if shuffle:
            random.shuffle(train_examples)

        return train_examples

    def save_model_and_copy_mcts(self, original: MCTS):
        """
        :param original: the MCTS to copy from
        :return: a fresh MCTS with a copy of the neural network inside
        """
        new_nnet = NeuralNetAdapter(input_size=17)
        original.nnet.save_checkpoint(folder=self.params.checkpoint_dir, filename='temperature.pth.tar')
        new_nnet.load_checkpoint(folder=self.params.checkpoint_dir, filename='temperature.pth.tar')
        new_mcts = MCTS(self.game, new_nnet, cpuct=original.cpuct, num_simulations=original.num_simulations)
        return new_mcts


    def train(self, defender: MCTS, challenger: MCTS):
        """
        Performs numIters iterations with numEps episodes of self-play in each
        iteration. After every iteration, it retrains neural network with
        examples in trajectories (which has a maximum length of maxlenofQueue).
        It then pits the new neural network against the old one and accepts it
        only if it wins >= updateThreshold fraction of games.
        """

        for iteration in range(1, self.params.num_iterations + 1):

            print(f"Iteraction {iteration} of {self.params.num_iterations+1}")

            # add some new examples, keep some old ones, drop the oldest
            trajectories = self.create_trajectories(defender, n_it=iteration)

            print("   Challenger to learn from the results")
            # training new network
            challenger.nnet.train(trajectories, params=self.params)

            # Arena!
            print('   Challenger meets Defender in the Arena')
            arena = Arena(lambda x: np.argmax(defender.get_action_prob(x, temperature=0)),
                          lambda x: np.argmax(challenger.get_action_prob(x, temperature=0)), self.game)
            defender_wins, challenger_wins, draws = arena.play_games(self.params.arena_compare)

            # Do we have a new champion?
            log.info('NEW/PREV WINS : %d / %d ; DRAWS : %d' % (challenger_wins, defender_wins, draws))
            if (defender_wins + challenger_wins == 0 or
                    float(challenger_wins) / (defender_wins + challenger_wins) < self.params.update_threshold):
                log.info('REJECTING NEW MODEL')
                challenger.nnet.load_checkpoint(folder=self.params.checkpoint_dir, filename='temperature.pth.tar')
            else:
                log.info('ACCEPTING NEW MODEL')
                challenger.nnet.save_checkpoint(folder=self.params.checkpoint_dir,
                                                filename=self.get_checkpoint_file(iteration))
                challenger.nnet.save_checkpoint(folder=self.params.checkpoint_dir,
                                                filename='best.pth.tar')

                defender = self.save_model_and_copy_mcts(challenger)

    # TODO: Make configurable or generally reconsider
    def get_checkpoint_file(self, iteration):
        return self.checkpoint_prefix + str(iteration) + '.pth.tar'


    def save_train_examples(self, iteration):
        folder = self.params.checkpoint_dir
        if not os.path.exists(folder):
            os.makedirs(folder)
        filename = os.path.join(folder, self.get_checkpoint_file(iteration) + ".examples")
        with open(filename, "wb+") as f:
            Pickler(f).dump(self.iterations_queue)

    def load_train_examples(self):
        model_file = os.path.join(self.params.load_folder_file[0], self.params.load_folder_file[1])
        examples_file = model_file + ".examples"
        if not os.path.isfile(examples_file):
            log.warning(f'File "{examples_file}" with trainExamples not found!')
            r = input("Continue? [y|n]")
            if r != "y":
                sys.exit()
        else:
            log.info("File with trainExamples found. Loading it...")
            with open(examples_file, "rb") as f:
                self.iterations_queue = Unpickler(f).load()
            log.info('Loading done!')

            # examples based on the model were already collected (loaded)
            self.skip_first_self_play = True


    def observe_trajectory(self, mcts: MCTS, verbose=False):

        board = self.game.get_initial_board()  # random start positions, sometimes unfair, but so what?
        if verbose:
            print(board)

        final_board = self.self_play(copy.deepcopy(board), mcts, verbose)

        examples = []
        start_at = len(board.get_stones())
        for stone in final_board.get_stones()[start_at:]:
            key = board.get_string_representation()
            probs = mcts.compute_probs(board, temperature=1.0)  # We explicitely want the temperature up, here
            q_advice = [mcts.Q.get((key, i), -float('inf')) for i in range(225)]
            v = np.max(q_advice, axis=None)

            sym = self.game.get_symmetries(board.canonical_representation(), probs)
            for b, p in sym:
                examples.append([b, p, v])
            board = copy.deepcopy(board)
            board.act(stone)

        return examples


    def self_play(self, board: Board, mcts: MCTS, verbose=False) -> Board:

        # Two mood versions of the champion playing against each other = less draws
        # These settings may change over the training period, once opponents get stronger.
        temperatures = [1, 0]  # more tight vs more explorative

        episode_step = 0
        done = self.game.get_game_ended(board)
        while done is None:
            episode_step += 1
            t = temperatures[episode_step % 2]
            pi = mcts.get_action_prob(board, temperature=t)
            action = np.random.choice(len(pi), p=pi)

            board.act(action)
            done = self.game.get_game_ended(board)
            if episode_step > 50:
                done = "draw"

            if verbose:
                print(board)

        # The player who made the last move, is the winner.
        if verbose:
            if done == 'draw':
                print("Draw")
            else:
                print(f"The winner is {1 - board.get_current_player()}")

        return board


    ## OLD

    def execute_episode_old(self, with_moves=False):
        """
        This function executes one episode of self-play, starting with player 1.
        As the game is played, each turn is added as a training example to
        train_examples. The game is played till the game ends. After the game
        ends, the outcome of the game is used to assign values to each example
        in train_examples.

        It uses a temperature=1 if episode_step < tempThreshold, and thereafter
        uses temperature=0.

        Returns:
            train_examples: a list of examples of the form (canonical_board, currPlayer, pi,v)
                           pi is the MCTS informed policy vector, v is +1 if
                           the player eventually won the game, else -1.
        """
        train_examples = []
        board = self.game.get_initial_board()
        self.current_player = board.get_current_player()
        episode_step = 0

        while True:
            episode_step += 1
            temperature = int(episode_step < self.params.temperature_threshold)

            pi = self.mcts.get_action_prob(board, temperature=temperature)
            sym = self.game.get_symmetries(board.canonical_representation(), pi)
            for b, p in sym:
                train_examples.append([b, self.current_player, p, None])

            action = np.random.choice(len(pi), p=pi)
            board, self.current_player = self.game.get_next_state(board, action)

            r = self.game.get_game_ended(board)

            if r is not None:
                train_examples = [(x[0], x[2], r * ((-1) ** (x[1] != self.current_player))) for x in train_examples]
                if with_moves:
                    return train_examples, self.recover_moves(train_examples)
                else:
                    return train_examples


    def train_old(self):
        """
        Performs numIters iterations with numEps episodes of self-play in each
        iteration. After every iteration, it retrains neural network with
        examples in train_examples (which has a maximum length of maxlenofQueue).
        It then pits the new neural network against the old one and accepts it
        only if it wins >= updateThreshold fraction of games.
        """

        defender = MCTS(self.game, self.nnet, self.params)

        for iteration in range(1, self.params.num_iterations + 1):

            # add some new examples, keep some old ones, drop the oldest
            train_examples = self.create_trajectories(n_it=iteration)

            # training new network
            self.nnet.train(train_examples)
            challenger = MCTS(self.game, self.nnet, self.params)

            # Arena!
            log.info('PITTING AGAINST PREVIOUS VERSION')
            arena = Arena(lambda x: np.argmax(defender.get_action_prob(x, temperature=0)),
                          lambda x: np.argmax(challenger.get_action_prob(x, temperature=0)), self.game)
            defender_wins, challenger_wins, draws = arena.play_games(self.params.arena_compare)

            # Do we have a new champion?
            log.info('NEW/PREV WINS : %d / %d ; DRAWS : %d' % (challenger_wins, defender_wins, draws))
            if (defender_wins + challenger_wins == 0 or
                    float(challenger_wins) / (defender_wins + challenger_wins) < self.params.update_threshold):
                log.info('REJECTING NEW MODEL')
                self.nnet.load_checkpoint(folder=self.params.checkpoint_dir, filename='temperature.pth.tar')
            else:
                log.info('ACCEPTING NEW MODEL')
                self.nnet.save_checkpoint(folder=self.params.checkpoint_dir,
                                          filename=self.get_checkpoint_file(iteration))
                self.nnet.save_checkpoint(folder=self.params.checkpoint_dir,
                                          filename='best.pth.tar')

                defender = self.copy_challenger()




    def create_trajectories_old(self, n_it: int):
        """
        :param n_it: the ordinal of the particular iteration
        :return: most recent training examples, containing some new and some old ones
            as a list of tuples (state, probs, value)
        """
        # examples of the iteration
        if not self.skip_first_self_play or n_it > 1:
            iteration = deque([], maxlen=self.params.max_queue_length)

            for _ in tqdm(range(self.params.num_episodes), desc="Self Play"):
                self.mcts = MCTS(self.game, self.nnet, self.params)  # reset search tree
                iteration += self.execute_episode()

            # save the iteration examples to the history
            self.iterations_queue.append(iteration)

        if len(self.iterations_queue) > self.params.num_iters_for_train_examples_history:
            log.warning(
                f"Removing the oldest entry in train_examples. len history = {len(self.iterations_queue)}")
            self.iterations_queue.pop(0)
        # backup history to a file
        # NB! the examples were collected using the model from the previous iteration, so (i-1)
        self.save_train_examples(n_it - 1)

        # shuffle examples before training
        train_examples = []
        for e in self.iterations_queue:
            train_examples.extend(e)
        shuffle(train_examples)

        return train_examples

