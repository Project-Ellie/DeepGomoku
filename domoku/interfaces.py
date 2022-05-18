import abc
from typing import Any, Optional, List


class Game(abc.ABC):
    """
    An ephemeral Game
    """
    @abc.abstractmethod
    def get_state(self: Any) -> Any:
        pass

    @abc.abstractmethod
    def do_move(self, move: Any) -> Any:
        pass

    @staticmethod
    @abc.abstractmethod
    def successor(state: Any, move: Any):
        """Return the (in determistic cases) or any legal (in stochastic cases) next state"""
        pass


class Player(abc.ABC):
    """
    An ephemeral player class just meant to encapsulate capabilities - not identies
    """
    def __init__(self):
        self.other = None

    @abc.abstractmethod
    def your_move(self, game: Game):
        """
        The player's definite move. No probabilistics
        :param game: any Game State
        :return: the players move
        """
        pass


class AbstractGanglion(abc.ABC):
    """
    A ganglion is some kind of AI that has the capabilities described by the abstrct methods herein.
    It may be a simple heuristic or a set of neural networks advising a complex tree search
    """

    @abc.abstractmethod
    def eval(self, state: Any) -> float:
        pass

    @abc.abstractmethod
    def winner(self, state: Any) -> Optional[int]:
        """
        :param state: any legal state
        :return: The integer index of the winning player, if the state is terminal, else None
        """
        pass

    @abc.abstractmethod
    def sample(self, state, n: int) -> List[Any]:
        """
        Compute n sample actions from that state
        :param state: any given state
        :param n: Any number of actions/moves to be independently drawn from the distribution represented by this.
        :return: a list of actions/moves
        """
        pass
