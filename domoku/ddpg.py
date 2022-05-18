from typing import Any

import numpy as np
from domoku import data
from domoku.interfaces import Player, Game, AbstractGanglion

MAX_TRY = 100  # general upper limit for anything that may fail a couple of times.


class NxNx4Game(Game):
    """
    A game representing the action on an N x N board with 4 channels
    """
    @staticmethod
    def successor(state: Any, move: Any):
        return data.after(state, move)

    def __init__(self, state: np.array):
        self._state = state

    def get_state(self) -> Any:
        return self._state

    def do_move(self, move: Any) -> Any:
        self._state = data.after(self._state, move)
        return self._state


class DdpgPlayer(Player):

    def __init__(self, brain: AbstractGanglion):
        super().__init__()
        self.brain = brain

    def your_move(self, game: Game) -> Any:
        return self.brain.sample(game.get_state(), 1)


class Trainer:
    """
    Here, we talk about players and games. But the interfaces are supposed to reflect samples and numbers
    """

    def __init__(self, heuristic: AbstractGanglion):
        self.heuristic = heuristic


    def create_trajectories(self, player1: Player, player2: Player,
                            num_trajectories, max_length, initial_state,
                            terminal: bool = True, verbose: bool = False):

        trajectories = []
        for _ in range(num_trajectories):
            trajectories.append(self.create_trajectory(player1, player2, max_length, initial_state, terminal, verbose))
        return trajectories


    def create_trajectory(self, player1: Player, player2: Player,
                          max_length, initial_state, terminal, verbose=False):
        """
        Create a trajectory from letting the two players play each other.
        Be aware that this method can fail and return None!
        :param player1 the first player to move
        :param player2 the second player to move
        :param max_length: Max length of the trajectory. In case it just doesn't terminate
        :param initial_state: the in
        :param terminal: whether or not the trajectory shall have a final terminal state
        :param verbose: print some diagnostics
        :return:
        """
        winning_channel = None
        for _ in range(MAX_TRY):
            trajectory = [(None, initial_state)]
            game = NxNx4Game(initial_state)

            player1.other, player2.other = player2, player1

            current_player = player1

            for _ in range(max_length):

                move = current_player.your_move(game)
                state = game.do_move(move)
                current_player = current_player.other

                if verbose:
                    from domoku import jupyter_tools as jt
                    jt.print_bin(state, combine=True)

                trajectory.append((move, state))
                winning_channel = self.heuristic.winner(state)
                if winning_channel is not None:
                    if verbose:
                        winning_color = data.get_winning_color(state, winning_channel)
                        winning_color = 'BLACK' if winning_color == 0 else 'WHITE'
                        print(f"{winning_color} wins.")
                    break
            if not terminal or (terminal and winning_channel is not None):
                return trajectory

            return None
