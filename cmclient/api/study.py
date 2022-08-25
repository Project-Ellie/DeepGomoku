from aegomoku.game_play import GamePlay
from aegomoku.gomoku_board import GomokuBoard
from aegomoku.interfaces import Move
from cmclient.api.basics import CompManConfig
from cmclient.gui import board


class StudyHandler:

    def __init__(self, config: CompManConfig = None):
        self.config = config
        self.current_state = ""
        self.game_play = GamePlay([])
        self.board = GomokuBoard(config.board_size)


    def handle(self):
        self.study()

    def study(self):
        board.show(registered="Self-Play Study", oppenent="",
                   move_listener=lambda move: self.move(move),
                   polling_listener=None,
                   board_size=self.config.board_size)

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

        return self.board
