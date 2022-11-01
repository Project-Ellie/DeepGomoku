from aegomoku.gomoku_game import ConstantBoardInitializer, \
    GomokuGame  # , RandomBoardInitializer, TopoSwap2BoardInitializer
from aegomoku.interfaces import MctsParams
from aegomoku.policies.heuristic_advice import HeuristicValueParams, HeuristicAdviser, HeuristicPolicyParams
from aegomoku.policies.heuristic_value_model import HeuristicValueModel
from cmclient.api.basics import CompManConfig
from cmclient.api.game_context import GameContext
from cmclient.api.study import StudyHandler

config = CompManConfig(board_size=19)


if __name__ == '__main__':
    # initializer = RandomBoardInitializer(config.board_size, 4, 9, 12, 9, 12)
    # initializer = TopoSwap2BoardInitializer(config.board_size)
    initializer = ConstantBoardInitializer("J10")  # "C11F9E9G8F7G7G9H8")
    model_file_name = None  # "DATA/models/4_c2s.model"
    import tensorflow as tf
    print(tf.config.list_physical_devices('GPU'))
    num_simulations = 2000

    game = GomokuGame(board_size=config.board_size, initializer=initializer)

    policy_params = HeuristicPolicyParams(board_size=config.board_size)
    value_params = HeuristicValueParams(config.board_size)
    value_model = HeuristicValueModel(value_params)
    adviser = HeuristicAdviser(policy_params, value_model)

    mcts_params = MctsParams(cpuct=0.2, temperature=0.0, num_simulations=num_simulations)
    context = GameContext(game, adviser, mcts_params)

    StudyHandler(context, config, "./cmclient/gui/", initializer).handle()
