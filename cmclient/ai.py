from aegomoku.gomoku_game import GomokuGame
from aegomoku.gomoku_players import PolicyAdvisedGraphSearchPlayer
from aegomoku.interfaces import MctsParams, PolicyParams


def get_player(game: GomokuGame, ai, num_simu):

    # TODO: Parameterize
    mcts = MctsParams(cpuct=1.0, temperature=0.0, num_simulations=num_simu)
    if ai is not None:
        policy = PolicyParams(model_file_name=f"DATA/models/{ai}", advice_cutoff=.01)
    else:
        policy = PolicyParams(model_file_name=None, advice_cutoff=.01)
    player = PolicyAdvisedGraphSearchPlayer("Policy-advised graph search", game, mcts, policy)
    return player, player.mcts, player.advisor
