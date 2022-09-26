from __future__ import annotations

from typing import Tuple, Optional

import numpy as np
import tensorflow as tf

from aegomoku.gomoku_board import PolicyAdviser
from aegomoku.interfaces import Player, Board, Move, MctsParams, PolicyParams, Game, Adviser
from aegomoku.mcts import MCTS
from aegomoku.policies.heuristic_policy import HeuristicPolicy


class PolicyAdvisedGraphSearchPlayer(Player):

    def __init__(self, name: str, game: Game, mcts_params: MctsParams,
                 policy_params: PolicyParams = None, adviser: Adviser = None):
        """
        :param name: Name of the player for logging and analysis
        :param game: the game, obviously
        :param mcts_params:
        :param policy_params: if provided, a PolicyAdvisor is created from these. Must be from a model file.
        :param adviser: if provided, that advisor is used, otherwise a Naive Heuristic Policy is created.
        """
        self.opponent: Optional[Player] = None
        self.name = name
        self.game = game
        self.mcts_params = mcts_params
        if policy_params is not None:
            if policy_params.model_file_name is not None:
                model = tf.keras.models.load_model(policy_params.model_file_name)
                self.advisor = PolicyAdviser(model=model, params=policy_params)
            else:
                raise ValueError("Must provide model file name")
        elif adviser is not None:
            self.advisor = adviser
        else:
            self.advisor = HeuristicPolicy(board_size=game.board_size, n_fwll=2)

        self.mcts = None
        self.refresh()

        super().__init__()


    def refresh(self):
        """
        resets all persistent state
        :return:
        """
        self.mcts = MCTS(self.game, self.advisor, self.mcts_params)


    def meet(self, other: Player):
        self.opponent = other
        other.opponent = self


    def move(self, board: Board, temperature=None, verbose=0) -> Tuple[Board, Optional[Move]]:
        """
        Procedural (not functional) interface. It changes the board!
        :param board: the board to use
        :param temperature: if provided, overrides the default training_data of the player. Good for self-play.
        :param verbose: Verbosity: 1=some, 2=lots
        :return: the very same board instance containing one more stone.
        """
        # No move when game is over
        winner = self.game.get_winner(board)
        if winner is not None:
            return board, winner

        temperature = temperature if temperature is not None else self.mcts.params.temperature
        probs = self.mcts.get_action_prob(board, temperature=temperature)

        patience = 5
        move = None

        while patience > 0:
            try:
                move = board.stone(np.random.choice(list(range(board.board_size**2 + 1)), p=probs))

            except Exception as e:
                print(f"Exception: {e}")
            if move not in board.get_stones():
                break
            patience -= 1

        if move is None:
            # TODO: Are there more reasonable alternatives?
            print("Truly sorry, but on...")
            print(board.get_stones())
            raise ArithmeticError("I can't find a possible move anymore")

        board.act(move)
        return board, move


    def evaluate(self, board: Board, temperature: float):
        key = board.get_string_representation()
        probs = self.mcts.compute_probs(board, temperature=temperature)
        q_advice = [self.mcts.Q.get((key, i)) for i in range(board.board_size * board.board_size)]
        q_advice = [q for q in q_advice if q is not None]
        v = 0.0 if len(q_advice) == 0 else np.max(q_advice, axis=None)
        return probs, v


    def __str__(self):
        return self.name

    __repr__ = __str__
