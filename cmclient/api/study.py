from aegomoku.game_play import GamePlay
from aegomoku.gomoku_board import GomokuBoard
from aegomoku.gomoku_game import GomokuGame, RandomBoardInitializer
from cmclient.api.basics import CompManConfig
from cmclient.api.game_context import GameContext
from cmclient.gui import board


class StudyHandler:

    def __init__(self, config: CompManConfig = None):
        self.config = config
        self.current_state = ""
        self.game_play = GamePlay([])
        self.board = GomokuBoard(config.board_size)
        initializer = RandomBoardInitializer(config.board_size, 4, 6, 8, 6, 8)
        self.game = GomokuGame(board_size=config.board_size, initializer=initializer)
        self.context = GameContext(self.game)
        self.ui = board.UI(game_context=self.context)


    def handle(self):
        self.ui.show(title="Self-Play Study")
