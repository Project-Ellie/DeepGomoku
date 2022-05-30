import logging
import os
import sys
from collections import deque
from pickle import Pickler, Unpickler
from random import shuffle

import numpy as np
from tqdm import tqdm

from alphazero.arena import Arena
from alphazero.interfaces import TrainParams
from alphazero.mcts import MCTS

log = logging.getLogger(__name__)


class Coach:
    """
    This class executes the self-play + learning. It uses the functions defined
    in Game and NeuralNet. params are specified in main.py.
    """

    def __init__(self, game, nnet, params: TrainParams):
        self.game = game
        self.nnet = nnet
        self.pnet = self.nnet.__class__(self.game)  # the competitor network
        self.params = params
        self.mcts = MCTS(self.game, self.nnet, self.params)
        self.train_examples_history = []  # history of examples from params.numItersForTrainExamplesHistory iterations
        self.skip_first_self_play = False  # can be overriden in load_train_examples()
        self.current_player = None
        self.checkpoint_prefix = 'checkpoint_'

    def execute_episode(self):
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
        self.current_player = 1
        episode_step = 0

        while True:
            episode_step += 1
            canonical_board = self.game.get_canonical_form(board, self.current_player)
            temperature = int(episode_step < self.params.temperature_threshold)

            pi = self.mcts.get_action_prob(canonical_board, temperature=temperature)
            sym = self.game.get_symmetries(canonical_board, pi)
            for b, p in sym:
                train_examples.append([b, self.current_player, p, None])

            action = np.random.choice(len(pi), p=pi)
            board, self.current_player = self.game.get_next_state(board, self.current_player, action)

            r = self.game.get_game_ended(board, self.current_player)

            if r != 0:
                return [(x[0], x[2], r * ((-1) ** (x[1] != self.current_player))) for x in train_examples]

    def learn(self):
        """
        Performs numIters iterations with numEps episodes of self-play in each
        iteration. After every iteration, it retrains neural network with
        examples in train_examples (which has a maximum length of maxlenofQueue).
        It then pits the new neural network against the old one and accepts it
        only if it wins >= updateThreshold fraction of games.
        """

        for i in range(1, self.params.num_iterations + 1):
            # bookkeeping
            log.info(f'Starting Iter #{i} ...')
            # examples of the iteration
            if not self.skip_first_self_play or i > 1:
                iteration_train_examples = deque([], maxlen=self.params.max_queue_length)

                for _ in tqdm(range(self.params.num_episodes), desc="Self Play"):
                    self.mcts = MCTS(self.game, self.nnet, self.params)  # reset search tree
                    iteration_train_examples += self.execute_episode()

                # save the iteration examples to the history 
                self.train_examples_history.append(iteration_train_examples)

            if len(self.train_examples_history) > self.params.num_iters_for_train_examples_history:
                log.warning(
                    f"Removing the oldest entry in train_examples. len history = {len(self.train_examples_history)}")
                self.train_examples_history.pop(0)
            # backup history to a file
            # NB! the examples were collected using the model from the previous iteration, so (i-1)  
            self.save_train_examples(i - 1)

            # shuffle examples before training
            train_examples = []
            for e in self.train_examples_history:
                train_examples.extend(e)
            shuffle(train_examples)

            # training new network, keeping a copy of the old one
            self.nnet.save_checkpoint(folder=self.params.checkpoint_dir, filename='temperature.pth.tar')
            self.pnet.load_checkpoint(folder=self.params.checkpoint_dir, filename='temperature.pth.tar')
            pmcts = MCTS(self.game, self.pnet, self.params)

            self.nnet.train(train_examples)
            nmcts = MCTS(self.game, self.nnet, self.params)

            log.info('PITTING AGAINST PREVIOUS VERSION')
            arena = Arena(lambda x: np.argmax(pmcts.get_action_prob(x, temperature=0)),
                          lambda x: np.argmax(nmcts.get_action_prob(x, temperature=0)), self.game)
            pwins, nwins, draws = arena.play_games(self.params.arena_compare)

            log.info('NEW/PREV WINS : %d / %d ; DRAWS : %d' % (nwins, pwins, draws))
            if pwins + nwins == 0 or float(nwins) / (pwins + nwins) < self.params.update_threshold:
                log.info('REJECTING NEW MODEL')
                self.nnet.load_checkpoint(folder=self.params.checkpoint_dir, filename='temperature.pth.tar')
            else:
                log.info('ACCEPTING NEW MODEL')
                self.nnet.save_checkpoint(folder=self.params.checkpoint_dir, filename=self.get_checkpoint_file(i))
                self.nnet.save_checkpoint(folder=self.params.checkpoint_dir, filename='best.pth.tar')

    # TODO: Make configurable or generally reconsider
    def get_checkpoint_file(self, iteration):
        return self.checkpoint_prefix + str(iteration) + '.pth.tar'

    def save_train_examples(self, iteration):
        folder = self.params.checkpoint_dir
        if not os.path.exists(folder):
            os.makedirs(folder)
        filename = os.path.join(folder, self.get_checkpoint_file(iteration) + ".examples")
        with open(filename, "wb+") as f:
            Pickler(f).dump(self.train_examples_history)

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
                self.train_examples_history = Unpickler(f).load()
            log.info('Loading done!')

            # examples based on the model were already collected (loaded)
            self.skip_first_self_play = True
