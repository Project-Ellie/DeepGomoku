from typing import List, Tuple, Any

from domoku.interfaces import AbstractGanglion, Game


class MinimaxSearch:

    def __init__(self, game: Game, policy: AbstractGanglion, max_depth, max_breadth):
        self.game = game
        self.policy = policy
        self.value_function = policy
        self.max_depth = max_depth
        self.max_breadth = max_breadth

    def minimax(self, state: Any, depth=None, alpha=-float('inf'), beta=float('inf'),
                is_max: bool = True, verbose: bool = False) -> Tuple[float, List[int]]:

        depth = self.max_depth if depth is None else depth

        if depth == 0 or self.policy.winner(state) is not None:
            value = self.value_function.eval(state)
            if verbose:
                print(f"Terminal state. Value = {value}")
            return value, []

        if verbose:
            print(f"M{'ax' if is_max else 'in'}imizing at depth: {depth}")

        moves = self.policy.sample(state, self.max_breadth)
        successors = {move: self.game.successor(state, move) for move in moves}

        if verbose:
            for key in successors:
                print(f"{key}: {successors[key]}")

        chosen = None
        chosen_history = []

        if is_max:
            max_val = -float('inf')
            for move, successor in successors.items():
                value, history = self.minimax(successor, depth-1, alpha, beta, False)
                if value > max_val:
                    chosen = move
                    chosen_history = history
                max_val = max(max_val, value)
                alpha = max(alpha, value)
                if beta <= alpha:
                    if verbose:
                        print("Pruning for Max")
                    return max_val, [chosen] + history

            if verbose:
                print(f"Chosen: {chosen}")
            return max_val, [chosen] + chosen_history

        else:
            min_val = float('inf')
            for move, successor in successors.items():
                value, history = self.minimax(successor, depth-1, alpha, beta, True)
                if value < min_val:
                    chosen = move
                    chosen_history = history
                min_val = min(min_val, value)
                beta = min(beta, value)
                if beta <= alpha:
                    if verbose:
                        print("Pruning for Min")
                    return min_val, [chosen]

            if verbose:
                print(f"Chosen: {chosen}")
            return min_val, [chosen] + chosen_history
