from aegomoku.game_play import GamePlay
from aegomoku.gomoku_board import GomokuBoard
from aegomoku.gomoku_game import GomokuGame
from aegomoku.interfaces import Move
from cmclient.ai import get_player


class GameContext:
    def __init__(self, game: GomokuGame, board_size: int):
        self.polling_listener = None
        self.game = game
        self.ai = get_player(board_size)
        self.game_play = GamePlay([])
        self.board = GomokuBoard(board_size)
        self.winner = None

    def move(self, stone: Move):
        stones = self.board.stones
        if len(stones) > 0 and stones[-1] == stone:
            self.board.remove(stone)
            self.game_play.bwd()
        else:
            if stone not in stones:
                self.game_play.fwd(stone.i)
                self.board.act(stone)
            else:
                # just ignore the rogue mouse click
                pass

        return self.board.get_stones()

    def ai_move(self):

        if self.game.get_game_ended(self.board) is not None:
            winner = (1 + len(self.board.get_stones())) % 2
            self.winner = winner
            self.ai = None
            return self.board.get_stones()

        self.ai.move(self.board)
        if self.game.get_game_ended(self.board) is not None:
            winner = (1 + len(self.board.get_stones())) % 2
            self.winner = winner
            self.ai = None

        return self.board.get_stones()

    def poll(self):
        return self.board.get_stones()
