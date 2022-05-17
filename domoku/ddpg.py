from domoku import data
from domoku.policies.heuristic import GomokuHeuristics

MAX_TRY = 100


class Trainer:

    def __init__(self, heuristic: GomokuHeuristics):
        self.policy_h = heuristic
        self.value_h = heuristic

    def create_trajectories(self, num_trajectories, max_length, initial_state, terminal: bool = True):
        trajectories = []
        for _ in range(num_trajectories):
            trajectories.append(self.create_trajectory(max_length, initial_state, terminal))
        return trajectories

    def create_trajectory(self, max_length, initial_state, terminal, verbose=False):
        """
        Be aware that this method can fail and return None!
        :param max_length: Max length of the trajectory. In case it just doesn't terminate
        :param initial_state: the in
        :param terminal: whether or not the trajectory shall have a final terminal state
        :param verbose: print some diagnostics
        :return:
        """
        winning_channel = None
        for _ in range(MAX_TRY):
            trajectory = [(None, initial_state)]
            state = initial_state
            move = self.policy_h.draw(state)
            for _ in range(max_length):

                state = data.after(state, move)
                if verbose:
                    from domoku import jupyter_tools as jt
                    jt.print_bin(state, combine=True)
                trajectory.append((move, state))
                winning_channel = self.policy_h.winner(state)
                if winning_channel is not None:
                    if verbose:
                        winning_color = data.get_winning_color(state, winning_channel)
                        winning_color = 'BLACK' if winning_color == 0 else 'WHITE'
                        print(f"{winning_color} wins.")
                    break
                move = self.policy_h.draw(state)
            if not terminal or (terminal and winning_channel is not None):
                return trajectory

            return None
