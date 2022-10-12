from aegomoku.gomoku_game import GomokuGame
from aegomoku.gomoku_players import PolicyAdvisedGraphSearchPlayer
from aegomoku.interfaces import MctsParams, PolicyParams


def get_player(game: GomokuGame, ai, num_simu):

    # TODO: Parameterize
    mcts_params = MctsParams(cpuct=1.0, temperature=0.0, num_simulations=num_simu)
    if ai is not None:
        policy_params = PolicyParams(model_file_name=ai, advice_cutoff=.01)
    else:
        policy_params = None
    player = PolicyAdvisedGraphSearchPlayer("Policy-advised graph search", game, mcts_params, policy_params)
    return player, player.mcts, player.adviser
