import abc
from typing import Any, Tuple


class GomokuHeuristics(abc.ABC):

    @abc.abstractmethod
    def value(self, state: Any) -> float:
        pass

    @abc.abstractmethod
    def winner(self, state) -> int:
        pass

    @abc.abstractmethod
    def draw(self, state) -> Tuple[int, int]:
        pass
