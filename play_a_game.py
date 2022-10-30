from aegomoku.advice import PolicyAdviser
from aegomoku.gomoku_game import ConstantBoardInitializer, \
    GomokuGame  # , RandomBoardInitializer, TopoSwap2BoardInitializer
from aegomoku.interfaces import MctsParams, PolicyParams
from aegomoku.policies.topological_value import TopologicalValuePolicy
from cmclient.api.basics import CompManConfig
from cmclient.api.game_context import GameContext
from cmclient.api.study import StudyHandler

config = CompManConfig(board_size=19)


def create_adviser(params: PolicyParams, board_size):
    if params.model_file_name is not None:
        model = tf.keras.models.load_model(params.model_file_name)
        return PolicyAdviser(model=model, params=params, board_size=board_size)
    else:
        return TopologicalValuePolicy(board_size=board_size,
                                      kappa_s=12.0, kappa_d=10.0,
                                      policy_stretch=2.0,
                                      value_stretch=1 / 32.,
                                      advice_cutoff=policy_params.advice_cutoff,
                                      noise_reduction=1.1,
                                      value_gauge=0.1)


if __name__ == '__main__':
    # initializer = RandomBoardInitializer(config.board_size, 4, 9, 12, 9, 12)
    initializer = ConstantBoardInitializer("C11F9E9G8F7G7G9H8I9H7I6I8J8")
    # initializer = TopoSwap2BoardInitializer(config.board_size)
    model_file_name = "DATA/models/4_c2s.model"
    import tensorflow as tf
    print(tf.config.list_physical_devices('GPU'))
    num_simulations = 1000

    mcts_params = MctsParams(cpuct=0.2, temperature=0.0, num_simulations=num_simulations)

    policy_params = PolicyParams(model_file_name=model_file_name, advice_cutoff=.01)
    adviser = create_adviser(policy_params,
                             config.board_size)

    game = GomokuGame(board_size=config.board_size, initializer=initializer)
    context = GameContext(game, mcts_params, policy_params, adviser)

    StudyHandler(context, config, "./cmclient/gui/", initializer).handle()
