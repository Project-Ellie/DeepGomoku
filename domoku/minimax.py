from typing import List, Tuple


class MinimaxSearch:

    def __init__(self, policy, value, max_depth, max_breadth):
        self.policy = policy
        self.value_function = value
        self.max_depth = max_depth
        self.max_breadth = max_breadth

    def minimax(self, state, depth, alpha, beta, is_max: bool, verbose: bool = False) -> Tuple[float, List[int]]:

        if depth == 0 or self.policy.is_terminated(state):
            if verbose:
                print(f"Terminal state. Value = {self.value_function.eval(state)}")
            return self.value_function.eval(state), []

        if verbose:
            print(f"M{'ax' if is_max else 'in'}imizing at depth: {depth}")

        moves = self.policy.sample(state)
        successors = {move: state.move(move) for move in moves}

        if verbose:
            print(f"Left : {successors[0]}")
            print(f"Right: {successors[1]}")

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
