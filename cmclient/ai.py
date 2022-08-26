from aegomoku.gomoku_game import GomokuGame, ConstantBoardInitializer
from aegomoku.gomoku_players import PolicyAdvisedGraphSearchPlayer
from aegomoku.interfaces import MctsParams, PolicyParams


def get_player(board_size: int):
    cbi = ConstantBoardInitializer("")
    gomoku_15x15 = GomokuGame(board_size=board_size, initializer=cbi)
    mcts = MctsParams(cpuct=1.0, temperature=.01, num_simulations=20)
    # policy = PolicyParams(model_file_name="models/first_model.model", advice_cutoff=.9)
    policy = PolicyParams(model_file_name=None, advice_cutoff=.9)
    player = PolicyAdvisedGraphSearchPlayer("Policy-advised graph search", gomoku_15x15, mcts, policy)
    return player
