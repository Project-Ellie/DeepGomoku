from aegomoku.interfaces import Board
from aegomoku.ray.trainer import StatefulRayPolicy
from aegomoku.policies.heuristic_policy import HeuristicPolicy


class HeuristicRayPolicy(StatefulRayPolicy):
    def init(self, *args, **kwargs):
        self.policy = HeuristicPolicy(*args, **kwargs)

    def get_winner(self, board: Board, **_):
        return self.policy.get_winner(board)

    def get_advisable_actions(self, state, **_):
        return self.policy.get_advisable_actions(state)

    def predict(self, board, **_):
        return self.policy.predict(board)

    def load_checkpoint(self, folder, filename, **_):
        raise NotImplementedError
