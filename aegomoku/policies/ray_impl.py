from aegomoku.interfaces import Board
from aegomoku.policies.topological_value import TopologicalValuePolicy
from aegomoku.ray.policy import StatefulRayPolicy
from aegomoku.policies.heuristic_policy import HeuristicPolicy


class HeuristicRayPolicy(StatefulRayPolicy):
    def init(self, *args, **kwargs):
        self.policy = TopologicalValuePolicy(*args, **kwargs)

    def get_winner(self, board: Board, **_):
        return self.policy.get_winner(board)

    def get_advisable_actions(self, state, **_):
        return self.policy.get_advisable_actions(state)

    def evaluate(self, board, **_):
        return self.policy.evaluate(board)

    def load_checkpoint(self, folder, filename, **_):
        raise NotImplementedError
