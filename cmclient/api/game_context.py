from aegomoku.game_play import GamePlay
from aegomoku.gomoku_game import GomokuGame
from aegomoku.interfaces import Move
from cmclient.ai import get_player


class GameContext:
    def __init__(self, game: GomokuGame, ai, num_simu):
        self.game = game
        self.num_simu = num_simu
        self.ai_spec = ai
        self.ai, self.mcts, self.advice = get_player(game, ai, num_simu)
        self.board = game.get_initial_board()
        self.game_play = GamePlay([s.i for s in self.board.stones])
        self.winner = None
        self.ai_active = True
        self.temperature = 1.0

    def get_advice(self):
        p_advice, p_value = self.advice.evaluate(self.board.canonical_representation())
        m_advice = self.mcts.compute_probs(self.board, self.temperature)
        key = self.board.get_string_representation()
        m_value = max([self.mcts.Q.get((key, i), -float('inf')) for i in range(225)])

        return (p_advice, p_value), (m_advice, m_value)

    def new_game(self):
        self.ai, self.mcts, self.advice = get_player(self.game, self.ai_spec, self.num_simu)
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

        return self.board.get_stones()

    def ai_move(self):

        if self.game.get_winner(self.board) is not None:
            winner = (1 + len(self.board.get_stones())) % 2
            self.winner = winner
            self.ai_active = False
            return self.board.get_stones()

        if self.ai is None:
            self.ai, self.mcts, self.advice = get_player(self.game, self.ai_spec, self.num_simu)

        _, move = self.ai.move(self.board)
        self.game_play.fwd(move)

        if self.game.get_winner(self.board) is not None:
            winner = (1 + len(self.board.get_stones())) % 2
            self.winner = winner
            self.ai_active = False

        return self.board.get_stones()

    def ponder(self, num_simulations: int = 1):
        self.mcts.ponder(self.board, num_simulations)
