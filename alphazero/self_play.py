import copy
import numpy as np
import ray

from alphazero.interfaces import Board, PolicySpec, Game, MctsParams
from alphazero.mcts import MCTS
from domoku.policies.heuristic_policy import HeuristicPolicy


@ray.remote
class SelfPlay:
    def __init__(self, mcts_params: MctsParams):
        self.mcts_params = mcts_params
        self.game = None
        self.mcts = None

    def init(self, board_size, game: Game, policy_spec: PolicySpec):

        self.game = game
        if policy_spec.type == PolicySpec.HEURISTIC:
            policy = HeuristicPolicy(board_size, cut_off=.1)
            policy.compile()
        elif policy_spec.type == PolicySpec.POOL_REF:
            policy = policy_spec.pool_ref
        else:
            raise ValueError("Only supporting heuristics for now.")
        self.mcts = MCTS(game, policy, self.mcts_params)


    def _self_play(self, board: Board, mcts: MCTS, verbose=False) -> Board:

        # Two mood versions of the champion playing against each other = less draws
        # These settings may change over the training period, once opponents get stronger.
        temperatures = [1, 0]  # more tight vs more explorative

        episode_step = 0
        done = self.game.get_game_ended(board)
        while done is None:
            episode_step += 1
            t = temperatures[episode_step % 2]
            pi = mcts.get_action_prob(board, temperature=t)
            action = np.random.choice(len(pi), p=pi)

            board.act(action)
            done = self.game.get_game_ended(board)
            if episode_step > 50:
                done = "draw"

            if verbose:
                print(board)

        # The player who made the last move, is the winner.
        if verbose:
            if done == 'draw':
                print("Draw")
            else:
                print(f"The winner is {1 - board.get_current_player()}")

        return board


    def _get_p_v(self, board: Board):
        key = board.get_string_representation()
        probs = self.mcts.compute_probs(board, temperature=1.0)  # We explicitely want the temperature up, here
        q_advice = [self.mcts.Q.get((key, i), -float('inf')) for i in range(225)]
        v = np.max(q_advice, axis=None)
        return probs, v


    def observe_trajectory(self, for_storage=False, verbose=False):

        board = self.game.get_initial_board()  # random start positions, sometimes unfair, but so what?
        if verbose:
            print(board)

        final_board = self._self_play(copy.deepcopy(board), self.mcts, verbose)
        stones = [stone.i for stone in final_board.get_stones()]

        if for_storage:
            data = []
            start_at = len(board.get_stones())
            for stone in final_board.get_stones()[start_at:]:
                position = [stone.i for stone in board.get_stones()]
                position = np.array(position).astype(np.uint8)
                probs, v = self._get_p_v(board)
                probs = (np.array(probs)*255).astype(np.uint8)
                data.append((position, probs, v))
                board = copy.deepcopy(board)
                board.act(stone)
            return data

        else:
            examples = []
            start_at = len(board.get_stones())

            for stone in final_board.get_stones()[start_at:]:
                probs, v = self._get_p_v(board)

                sym = self.game.get_symmetries(board.canonical_representation(), probs)
                for b, p in sym:
                    examples.append([b, p, v])
                board = copy.deepcopy(board)
                board.act(stone)

            return examples, stones
