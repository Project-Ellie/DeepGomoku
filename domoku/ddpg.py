from typing import Any

from domoku.policies.maximal_criticality import MaxCriticalityPolicy
from domoku.policies.softadvice import MaxInfluencePolicy, MaxInfluencePolicyParams
from domoku import data
from domoku.interfaces import Player, Game, AbstractGanglion
from domoku.minimax import MinimaxSearch
from domoku.constants import *
from domoku.tools import GomokuTools as gt

MAX_TRY = 100  # general upper limit for anything that may fail a couple of times.


class NxNx4Game(Game):
    """
    A game representing the action on an N x N board with 4 channels
    """

    def __init__(self, state: np.array = None):
        self._state = state
        if state is not None:
            self.n = state.shape[0]


    def as_str(self, move):
        """
        :param move: a move as row/column of a matrix
        :return: the string rep of that move like eg H13, etch
        """
        x, y = gt.m2b(move, self.n)
        return f"{chr(x + 64)}{y}"


    @staticmethod
    def successor(state: Any, move: Any):
        return data.after(state, move)


    def get_state(self) -> Any:
        return self._state


    def do_move(self, move: Any) -> Any:
        self._state = data.after(self._state, move)
        return self._state


class DdpgPlayer(Player):

    def __init__(self, brain: AbstractGanglion, tree_search=None):
        self.tree_search = tree_search if tree_search is not None else None
        super().__init__()
        self.brain = brain
        self.tree_search = tree_search


    def your_action(self, game: Game) -> Any:
        if self.tree_search is not None:
            search_depth, search_width = self.tree_search

            search = MinimaxSearch(NxNx4Game(game.get_state()), self.brain,
                                   search_depth, search_width)
            value, moves = search.minimax(game.get_state())
            return moves[0]
        else:
            return self.brain.sample(game.get_state(), 1)[0]


    def your_move(self, game: NxNx4Game) -> Move:
        r, c = gt.m2b(self.your_action(game), game.n)
        return Move(r, c)


class Trainer:
    """
    Here, we talk about players and games. But the interfaces are supposed to reflect samples and numbers
    """


    def __init__(self, board_size: int, heuristic: AbstractGanglion):
        self.board_size = board_size
        self.heuristic = heuristic
        self.criticality_model = MaxCriticalityPolicy(board_size, overconfidence=2)
        self.black = self.create_default_player(point_of_view=BLACK)
        self.white = self.create_default_player(point_of_view=WHITE)


    def create_trajectories(self, num_trajectories, max_length, initial_state,
                            black: Player = None, white: Player = None,
                            terminal: bool = True, verbose: bool = False):

        trajectories = []

        black = black if black is not None else self.black
        white = white if white is not None else self.white

        for _ in range(num_trajectories):
            trajectories.append(self.create_trajectory(max_length, initial_state, terminal, black, white, verbose))
        return trajectories


    def create_trajectory(self, max_length, initial_state, terminal,
                          black: Player = None, white: Player = None,
                          verbose=False):
        """
        Create a trajectory from letting the two players play each other.
        Be aware that this method can fail and return None!
        :param black: the player playing black
        :param white the playere playing white
        :param max_length: Max length of the trajectory. In case it just doesn't terminate
        :param initial_state: the in
        :param terminal: whether or not the trajectory shall have a final terminal state
        :param verbose: print some diagnostics
        :return: a game play trajectory
        """

        if np.sum(initial_state, axis=None) % 2 == 1:
            first, second = white, black
        else:
            first, second = black, white

        winning_channel = None
        for _ in range(MAX_TRY):
            trajectory = [(None, initial_state)]
            game = NxNx4Game(initial_state)

            first.other, second.other = second, first

            current_player = first

            for _ in range(max_length):

                move = current_player.your_action(game)
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


    def create_default_player(self, point_of_view, tree_search=None):
        params = MaxInfluencePolicyParams(
            board_size=self.board_size,
            sigma=.7,
            iota=10,
            radial_constr=[.0625, .125, .25, .5],
            radial_obstr=[-.0625, -.125, -.25, -.5]
        )
        heuristics = MaxInfluencePolicy(params, pov=point_of_view, criticality_model=self.criticality_model)
        return DdpgPlayer(heuristics, tree_search)
