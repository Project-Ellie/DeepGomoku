from __future__ import annotations

from typing import Tuple, Optional

import numpy as np

from alphazero.interfaces import Player, Board, Move
from alphazero.mcts import MCTS


class HeuristicPlayer(Player):

    def __init__(self, name: str, mcts: MCTS, temperature: float):
        """
        :param mcts: The search tree to use
        :param temperature: a float between 1.0 for more exploration and 0.0 for only the best move to take
        """
        self.opponent: Optional[Player] = None
        self.name = name
        self.mcts = mcts
        self.temperature = temperature
        super().__init__()


    def meet(self, other: Player):
        self.opponent = other
        other.opponent = self


    def move(self, board: Board, temperature=None) -> Tuple[Board, Move]:
        """
        Procedural (not functional) interface. It changes the board!
        :param board: the board to use
        :param temperature: if provided, overrides the default temperature of the player. Good for self-play.
        :return: the very same board instance containing one more stone.
        """
        temperature = temperature if temperature is not None else self.temperature
        probs = self.mcts.get_action_prob(board, temperature=temperature)
        move = board.stone(np.random.choice(225, p=probs))
        board.act(move)
        return board, move


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
