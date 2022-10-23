
from aegomoku.game_play import GamePlay
from aegomoku.gomoku_board import GomokuBoard
from aegomoku.gomoku_game import GomokuGame, BoardInitializer
import aegomoku.tools as gt
from cmclient.api.basics import CompManConfig
from cmclient.api.game_context import GameContext
from cmclient.gui import board


class StudyHandler:

    def __init__(self, config: CompManConfig = None, base_path=None,
                 initializer: BoardInitializer = None, ai=None, num_simu=100):
        self.config = config
        self.current_state = initializer.initial_stones()
        self.board = GomokuBoard(config.board_size, stones=self.current_state)
        stones = gt.string_to_stones(self.current_state) if self.current_state != "" else []
        self.game_play = GamePlay([self.board.Stone(*coords).i for coords in stones])
        self.game = GomokuGame(board_size=config.board_size, initializer=initializer)
        self.context = GameContext(self.game, ai, num_simu)
        self.ui = board.UI(game_context=self.context, base_path=base_path)


    def handle(self):
        self.ui.show(title="Self-Play Study")
