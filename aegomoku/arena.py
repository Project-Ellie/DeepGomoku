import logging
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
        :param player1: The player to move first
        :param player2: the other player
        :param game: the game
        :param max_moves: the max number of moves. Failing to terminate before makes the game a draw
        :param display: a callable for debugging purposes.
        """
        self.player1 = player1
        self.player2 = player2
        player1.meet(player2)
        self.game = game
        self.display = display
        self.max_moves = max_moves
        self.board = None  # for reference after the match


    def play_game(self, switch=False, verbose=False) -> Optional[Player]:
        """
        :param switch: if True, player2 will make the first move
        :param verbose: debug info
        :return:
        """
        self.board = board = self.game.get_initial_board()
        if verbose:
            first_color = _current_color(self.board)
            print(f"{self.player1.name} to begin with {first_color}.")

        n_moves = 0
        # The player will change a last time before the first move
        player = self.player2
        if switch:
            player = player.opponent

        while self.game.get_game_ended(board) is None and n_moves < self.max_moves:
            player = player.opponent
            _, move = player.move(board)
            if verbose:
                print(f"{player.name}: {move}")
            if verbose > 1:
                board.plot()
        if verbose:
            print(f"{player.name} ({_previous_color(board)}) won.")
        return player


    def play_games(self, num, verbose=False):
        """
        Plays num games in which player1 starts num/2 games and player2 starts
        num/2 games.

        Returns: a dictionary with the stots for player1, player2 and draws
        """

        num = int(num / 2)
        stats = {self.player1: 0,
                 self.player2: 0,
                 "draws": 0}

        for _ in tqdm(range(num), desc="Arena.play_games (1)"):
            winner = self.play_game(verbose=verbose)
            self._update_stats(stats, winner)

        for _ in tqdm(range(num), desc="Arena.play_games (2)"):
            winner = self.play_game(switch=True, verbose=verbose)
            self._update_stats(stats, winner)

        return stats

    @staticmethod
    def _update_stats(stats: Dict, winner: Player):
        if winner is None:
            stats['draws'] += 1
        else:
            stats[winner] += 1