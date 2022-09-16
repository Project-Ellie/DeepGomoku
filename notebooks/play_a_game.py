from aegomoku.gomoku_game import ConstantBoardInitializer, RandomBoardInitializer
from cmclient.api.basics import CompManConfig
from cmclient.api.study import StudyHandler

config = CompManConfig(board_size=19)

if __name__ == '__main__':
    #  initializer = RandomBoardInitializer(config.board_size, 4, 6, 8, 6, 8)
    initializer = ConstantBoardInitializer("")
    ai = "3_c2.model"
    num_simulations = 600
    StudyHandler(config, "../cmclient/gui/", initializer, ai, num_simu=num_simulations).handle()
