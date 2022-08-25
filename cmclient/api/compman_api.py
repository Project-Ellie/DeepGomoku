from cmclient.api.basics import CompManConfig, PLAYERS, CONFIG, TABLES, GAMES, AbstractCompManApi, STUDY
from cmclient.api.config import ConfigHandler
from cmclient.api.games import GameHandler
from cmclient.api.players import PlayerHandler, PlayerApi
from cmclient.api.rest import RestAdapter
from cmclient.api.study import StudyHandler
from cmclient.api.tables import TableHandler, TableApi


class CompManApi(PlayerApi, TableApi, AbstractCompManApi):
    pass


class CompManHandler:

    def __init__(self, adapter: RestAdapter, config: CompManConfig):

        self.adapter = RestAdapter(adapter, config.host)

        superapi = CompManApi(adapter, config)

        self.handlers = {
            PLAYERS: PlayerHandler(adapter, config),
            TABLES: TableHandler(adapter, config),
            CONFIG: ConfigHandler(adapter, config),
            GAMES: GameHandler(adapter, config, superapi),
            STUDY: StudyHandler(config)
        }

    def handle(self, args):
        output = self.handlers[args.context].handle(args)
        return output
