from aegomoku.advice import PolicyAdviser
from aegomoku.gomoku_game import ConstantBoardInitializer, \
    GomokuGame  # , RandomBoardInitializer, TopoSwap2BoardInitializer
from aegomoku.interfaces import MctsParams, PolicyParams
from aegomoku.policies.heuristic_advice import HeuristicAdviser, HeuristicAdviserParams
from aegomoku.policies.topological_value import TopologicalValuePolicy
from cmclient.api.basics import CompManConfig
from cmclient.api.game_context import GameContext
from cmclient.api.study import StudyHandler

config = CompManConfig(board_size=19)


def create_adviser(params: PolicyParams):
    if params.model_file_name is not None:
        model = tf.keras.models.load_model(params.model_file_name)
        return PolicyAdviser(model=model, params=params)
    else:
        params = HeuristicAdviserParams(board_size=params.board_size,
                                        advice_threshold=.02,
                                        criticalities=None,
                                        min_secondary=5,
                                        percent_secondary=0)
        return HeuristicAdviser(params)


def creata_topo_adviser(params: PolicyParams):
    return TopologicalValuePolicy(board_size=params.board_size,
                                  kappa_s=12.0, kappa_d=10.0,
                                  policy_stretch=2.0,
                                  value_stretch=1 / 32.,
                                  advice_cutoff=policy_params.advice_cutoff,
                                  noise_reduction=1.1,
                                  value_gauge=0.1)


if __name__ == '__main__':
    # initializer = RandomBoardInitializer(config.board_size, 4, 9, 12, 9, 12)
    # initializer = TopoSwap2BoardInitializer(config.board_size)
    initializer = ConstantBoardInitializer("C11F9E9G8F7G7G9H8")
    model_file_name = None  # "DATA/models/4_c2s.model"
    import tensorflow as tf
    print(tf.config.list_physical_devices('GPU'))
    num_simulations = 2000

    mcts_params = MctsParams(cpuct=0.2, temperature=0.0, num_simulations=num_simulations)

    policy_params = PolicyParams(board_size=config.board_size,
                                 model_file_name=model_file_name, advice_cutoff=.01)
    adviser_factory = create_adviser

    game = GomokuGame(board_size=config.board_size, initializer=initializer)
    context = GameContext(game, mcts_params, policy_params, adviser_factory)

    StudyHandler(context, config, "./cmclient/gui/", initializer).handle()
