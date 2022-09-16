import logging
from pickle import Pickler
from typing import Optional, Callable, Dict

from tqdm import tqdm

from aegomoku.interfaces import Player, Game

log = logging.getLogger(__name__)


def _current_color(board) -> str:
    return 'white' if board.get_current_player() == 1 else 'black'


def _previous_color(board) -> str:
    return 'white' if board.get_current_player() == 0 else 'black'


class Arena:
    def __init__(self, player1: Player, player2: Player, game: Game, max_moves: int,
                 display: Optional[Callable] = None):
        """
        :param: player1: The player to move first
        :param player2: the other player
        :param game: the game
        :param max_moves: the max number of moves. Failing to terminate before makes the game a draw
        :param display: a callable for debugging purposes.
        """
        self.games = []
        self.player1 = player1
        self.player2 = player2
        player1.meet(player2)
        self.game = game
        self.display = display
        self.max_moves = max_moves
        self.board = None  # for reference after the match


    def play_game(self, switch=False, verbose=0) -> Optional[Player]:
        """
        :param switch: if True, player2 will make the first move
        :param verbose: provide debug info
        :return:
        """
        self.board = board = self.game.get_initial_board()

        n_moves = 0
        # The player will change a last time before the first move
        player = self.player2
        if switch:
            player = player.opponent

        if verbose:
            first_color = _current_color(self.board)
            print(f"{player.opponent.name} to begin with {first_color}.")

        while self.game.get_winner(board) is None and n_moves < self.max_moves:
            player = player.opponent
            _, move = player.move(board)
            if verbose > 1:
                print(f"{n_moves + 1}: {player.name}: {move}")
            if verbose > 2:
                board.plot()
                print(board.get_stones())
                print()
            n_moves += 1
        if verbose and n_moves < self.max_moves:
            print(f"{player.name} ({_previous_color(board)}) won.")
            return player
        else:
            print(f"Draw after {n_moves} moves.")
            return None


    def play_games(self, num, verbose=0, save_to=None):
        """
        Plays num games in which player1 starts num/2 games and player2 starts
        num/2 games.

        Returns: a dictionary with the stots for player1, player2 and draws
        """

        num = int(num / 2)
        stats = {self.player1: 0,
                 self.player2: 0,
                 "draws": 0}

        for nr in range(num):
            try:
                if verbose > 0:
                    print(f"Game {nr + 1}: ", end='')
                winner = self.play_game(verbose=verbose)
                self.games.append([stone.i for stone in self.board.stones])
                self._update_stats(stats, winner)
                if verbose > 0:
                    print([self.board.Stone(i) for i in self.games[-1]])
            except Exception as e:
                print(e)
                "Simply ignoring this game."

        for nr in range(num):
            try:
                if verbose > 0:
                    print(f"Game {num + nr + 1}: ", end='')
                winner = self.play_game(switch=True, verbose=verbose)
                self.games.append([stone.i for stone in self.board.stones])
                self._update_stats(stats, winner)
                if verbose > 0:
                    print([self.board.Stone(i) for i in self.games[-1]])
            except Exception as e:
                print(e)
                "Simply ignoring this game."

        if save_to is not None:

            with open(save_to, 'wb+') as f:
                Pickler(f).dump(self.games)
                if verbose > 0:
                    print(f"Saved gameplay data to {save_to}")

        return stats

    @staticmethod
    def _update_stats(stats: Dict, winner: Player):
        if winner is None:
            stats['draws'] += 1
        else:
            stats[winner] += 1
