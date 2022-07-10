import logging
from typing import Optional, Callable, Dict

from tqdm import tqdm

from alphazero.interfaces import Player, Game

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


# Legacy - remove once done
class OldArena:
    """
    An Arena class where any 2 agents can be pitted against each other.
    """

    def __init__(self, player1, player2, game, display=None):
        """
        Input:
            player 1,2: two Players
            game: Game object
            display: a function that takes board as input and prints it (e.g.
                     display in othello/OthelloGame). Is necessary for verbose
                     mode.

        see othello/OthelloPlayers.py for an example. See pit.py for pitting
        human players/other baselines with each other.
        """
        self.player1 = player1
        self.player2 = player2
        self.game = game
        self.display = display

    def play_game(self, verbose=False):
        """
        Executes one episode of a game.

        Returns:
            either
                winner: player who won the game (1 if player1, -1 if player2)
            or
                draw result returned from the game that is neither 1, -1, nor 0.
        """
        players = [self.player2, None, self.player1]
        current_player = 1
        board = self.game.get_initial_board()
        it = 0
        while self.game.get_game_ended(board) is None:
            it += 1
            if verbose:
                assert self.display
                print("Turn ", str(it), "Player ", str(current_player))
                self.display(board)
            action = players[current_player + 1](board.canonical_representation())

            valids = self.game.get_valid_moves(self.game.get_canonical_form(board, current_player), 1)

            if valids[action] == 0:
                log.error(f'Action {action} is not valid!')
                log.debug(f'valids = {valids}')
                assert valids[action] > 0
            board, current_player = self.game.get_next_state(board, current_player, action)
        if verbose:
            assert self.display
            print("Game over: Turn ", str(it), "Result ", str(self.game.get_game_ended(board)))
            self.display(board)
        return current_player * self.game.get_game_ended(board, current_player)

    def play_games(self, num, verbose=False):
        """
        Plays num games in which player1 starts num/2 games and player2 starts
        num/2 games.

        Returns:
            one_won: games won by player1
            two_won: games won by player2
            draws:  games won by nobody
        """

        num = int(num / 2)
        one_won = 0
        two_won = 0
        draws = 0
        for _ in tqdm(range(num), desc="Arena.play_games (1)"):
            game_result = self.play_game(verbose=verbose)
            if game_result == 1:
                one_won += 1
            elif game_result == -1:
                two_won += 1
            else:
                draws += 1

        self.player1, self.player2 = self.player2, self.player1

        for _ in tqdm(range(num), desc="Arena.play_games (2)"):
            game_result = self.play_game(verbose=verbose)
            if game_result == -1:
                one_won += 1
            elif game_result == 1:
                two_won += 1
            else:
                draws += 1

        return one_won, two_won, draws
