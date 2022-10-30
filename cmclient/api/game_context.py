from aegomoku.game_play import GamePlay
from aegomoku.gomoku_game import GomokuGame
from aegomoku.gomoku_players import PolicyAdvisedGraphSearchPlayer
from aegomoku.interfaces import Move


class GameContext:
    def __init__(self, game: GomokuGame,
                 mcts_params, policy_params, adviser):
        self.game = game
        self.mcts_params = mcts_params
        self.board = game.get_initial_board()
        self.game_play = GamePlay([s.i for s in self.board.stones])
        self.winner = None
        self.ai_active = True
        self.temperature = 1.0
        self.player = PolicyAdvisedGraphSearchPlayer(game, mcts_params, policy_params, adviser)
        self.adviser = adviser

    def get_advice(self):
        state = self.board.canonical_representation()
        _, p_value = self.adviser.evaluate(state)
        p_advice = self.adviser.advise(state)
        m_advice = self.player.mcts.compute_probs(self.board, self.temperature)
        key = self.board.get_string_representation()

        m_value = max([self.player.mcts.Q.get((key, i), -float('inf')) for i in range(self.board.board_size)])

        return (p_advice, p_value), (m_advice, m_value)

    def new_game(self):
        self.player.refresh()
        self.adviser = self.player.adviser
        self.board = self.game.get_initial_board()
        self.game_play = GamePlay([stone.i for stone in self.board.stones])
        self.winner = None
        return self.board.get_stones()

    def bwd(self):
        # Deactivate AI when move is withdrawn
        stones = self.board.get_stones()
        if len(stones) > 0:
            self.board.remove(stones[-1])
        self.game_play.bwd()
        self.winner = None
        self.ai_active = False
        return self.board.get_stones()

    def fwd(self):
        current = self.game_play.fwd()
        if current is None:
            return self.board.get_stones()
        stones = [stone.i for stone in self.board.get_stones()]
        if current.move not in stones:
            self.board.act(current.move)
        return self.board.get_stones()

    def move(self, stone: Move):
        stones = self.board.stones

        if len(stones) > 0 and stones[-1] == stone:
            return self.bwd()
        else:
            if stone in stones:
                # just ignore the rogue mouse click on any but the last existing stone
                return None
            else:
                # create a new branch to study
                self.game_play.fwd(stone.i)
                self.board.act(stone)

        self.dump_game()
        return self.board.get_stones()

    def ai_move(self):

        if self.game.get_winner(self.board) is not None:
            winner = (1 + len(self.board.get_stones())) % 2
            self.winner = winner
            self.ai_active = False
            return self.board.get_stones()

        _, move = self.player.move(self.board)
        self.game_play.fwd(move)

        if self.game.get_winner(self.board) is not None:
            winner = (1 + len(self.board.get_stones())) % 2
            self.winner = winner
            self.ai_active = False

        self.dump_game()
        return self.board.get_stones()

    def dump_game(self):
        stones = "".join([str(stone) for stone in self.board.get_stones()])
        with open("aegomoku.data", 'w') as f:
            f.write(stones)


    def ponder(self, num_simulations: int = 1):
        self.player.mcts.ponder(self.board, num_simulations)
