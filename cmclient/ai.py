from aegomoku.gomoku_game import GomokuGame
from aegomoku.gomoku_players import PolicyAdvisedGraphSearchPlayer
from aegomoku.interfaces import MctsParams, PolicyParams


def get_player(game: GomokuGame):

    # TODO: Parameterize
    mcts = MctsParams(cpuct=4.0, temperature=0.0, num_simulations=1200)
    # policy = PolicyParams(model_file_name="models/first_model.model", advice_cutoff=.2)
    policy = PolicyParams(model_file_name=None, advice_cutoff=.01)
    player = PolicyAdvisedGraphSearchPlayer("Policy-advised graph search", game, mcts, policy)
    return player, player.mcts, player.advisor
