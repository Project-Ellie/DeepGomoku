from __future__ import annotations

from typing import Tuple, Optional

import numpy as np
import tensorflow as tf
from aegomoku.interfaces import Player, Board, Move, MctsParams, PolicyParams, PolicyAdviser, Game, Adviser
from aegomoku.mcts import MCTS
from aegomoku.policies.topological_value import TopologicalValuePolicy


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
                self.adviser = PolicyAdviser(model=model, params=policy_params, board_size=game.board_size)
            else:
                raise ValueError("Must provide model file name")
        elif adviser is not None:
            self.adviser = adviser
        else:
            self.adviser = TopologicalValuePolicy(self.game.board_size,
                                                  percent_secondary=0,
                                                  min_secondary=0)

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


# Legacy players - remove one day
class RandomPlayer:
    def __init__(self, game):
        self.game = game

    def play(self, board):
        a = np.random.randint(self.game.get_action_size())
        valids = self.game.get_valid_moves(board, 1)
        while valids[a] != 1:
            a = np.random.randint(self.game.get_action_size())
        return a


class HumanGobangPlayer:
    def __init__(self, game):
        self.game = game

    def play(self, board):
        # display(board)
        valid = self.game.get_valid_moves(board, 1)
        for i in range(len(valid)):
            if valid[i]:
                print(int(i/self.game.n), int(i % self.game.n))
        while True:
            a = input()

            x, y = [int(x) for x in a.split(' ')]
            a = self.game.n * x + y if x != -1 else self.game.n ** 2
            if valid[a]:
                break
            else:
                print('Invalid')

        return a


class GreedyGobangPlayer:
    def __init__(self, game):
        self.game = game

    def play(self, board):
        valids = self.game.get_valid_moves(board, 1)
        candidates = []
        for a in range(self.game.get_action_size()):
            if valids[a] == 0:
                continue
            next_board, _ = self.game.get_next_state(board, 1, a)
            score = self.game.getScore(next_board, 1)
            candidates += [(-score, a)]
        candidates.sort()
        return candidates[0][1]
