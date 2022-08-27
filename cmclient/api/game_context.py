from aegomoku.game_play import GamePlay
from aegomoku.gomoku_board import GomokuBoard
from aegomoku.gomoku_game import GomokuGame
from aegomoku.interfaces import Move
from cmclient.ai import get_player


class GameContext:
    def __init__(self, game: GomokuGame):
        self.game = game
        self.ai, self.mcts, self.advice = get_player(game)
        self.game_play = GamePlay([])
        self.board = GomokuBoard(game.board_size)
        self.winner = None
        self.ai_active = True

    def get_advice(self):
        policy_advice = self.advice.evaluate(self.board.canonical_representation())
        return policy_advice, (None, None)

    def new_game(self):
        self.ai, self.mcts, self.advice = get_player(self.game)
        self.board = self.game.get_initial_board()
        self.game_play = GamePlay([stone.i for stone in self.board.stones])
        self.winner = None
        return self.board.get_stones()

    def bwd(self):
        # Deactivate AI when move is withdrawn
        stones = self.board.get_stones()
        self.board.remove(stones[-1])
        self.game_play.bwd()
        self.winner = None
        self.ai_active = False
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

        if self.game.get_game_ended(self.board) is not None:
            winner = (1 + len(self.board.get_stones())) % 2
            self.winner = winner
            self.ai_active = False
            return self.board.get_stones()

        if self.ai is None:
            self.ai, self.mcts, self.advice = get_player(self.game)

        self.ai.move(self.board)

        if self.game.get_game_ended(self.board) is not None:
            winner = (1 + len(self.board.get_stones())) % 2
            self.winner = winner
            self.ai_active = False

        return self.board.get_stones()
