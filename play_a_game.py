from aegomoku.gomoku_game import ConstantBoardInitializer, RandomBoardInitializer, TopoSwap2BoardInitializer
from cmclient.api.basics import CompManConfig
from cmclient.api.study import StudyHandler

config = CompManConfig(board_size=19)

if __name__ == '__main__':
    # initializer = RandomBoardInitializer(config.board_size, 4, 9, 12, 9, 12)
    initializer = ConstantBoardInitializer("J10")
    # initializer = TopoSwap2BoardInitializer(config.board_size)
    ai = "3_c2.model"
    num_simulations = 400
    StudyHandler(config, "./cmclient/gui/", initializer, ai, num_simu=num_simulations).handle()
