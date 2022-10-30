from __future__ import annotations

from typing import Tuple, Optional

import numpy as np

from aegomoku.interfaces import Player, Board, Move, MctsParams, PolicyParams, Game
from aegomoku.mcts import MCTS


class PolicyAdvisedGraphSearchPlayer(Player):

    def __init__(self, game: Game, adviser_factory,
                 mcts_params: MctsParams,
                 policy_params: PolicyParams = None, name=None):
        """
        :param name: Name of the player for logging and analysis
        :param game: the game, obviously
        :param mcts_params:
        :param policy_params: if provided, a PolicyAdvisor is created from these. Must be from a model file.
        :param adviser_factory: A method that takes the policyParams and returns an Adviser instance
        """
        self.opponent: Optional[Player] = None
        self.name = name
        self.game = game
        self.adviser_factory = adviser_factory
        self.mcts_params = mcts_params

        self.adviser = adviser_factory(policy_params)

        self.mcts = MCTS(self.game, self.adviser, self.mcts_params)
        self.refresh()

        super().__init__()


    def refresh(self):
        """
        resets all persistent state
        :return:
        """
        self.mcts = MCTS(self.game, self.adviser, self.mcts_params)


    def meet(self, other: Player):
        self.opponent = other
        other.opponent = self


    def move(self, board: Board, temperature=None) -> Tuple[Board, Optional[Move]]:
        """
        Procedural (not functional) interface. It changes the board!
        :param board: the board to use
        :param temperature: if provided, overrides the default training_data of the player. Good for self-play.
        :return: the very same board instance containing one more stone.
        """
        # No move when game is over
        winner = self.game.get_winner(board)
        if winner is not None:
            return board, None

        temperature = temperature if temperature is not None else self.mcts.params.temperature
        probs = self.mcts.get_action_prob(board, temperature=temperature)

        patience = 5
        move = None

        while patience > 0:
            move = board.stone(np.random.choice(list(range(board.board_size**2)), p=probs))
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
