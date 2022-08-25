from cmclient.api.basics import CompManConfig, PLAYERS, TABLES
from cmclient.api.compman_api import CompManHandler
from cmclient.api.players import PlayerHandler
from cmclient.api.rest import RestAdapter
from cmclient.api.tables import TableHandler


def get_player_handler(client) -> PlayerHandler:
    config = CompManConfig(host="")
    adapter = RestAdapter(client, config.host)
    api = CompManHandler(adapter, config=config)
    return api.handlers[PLAYERS]


def get_table_handler(client) -> TableHandler:
    config = CompManConfig(host="")
    adapter = RestAdapter(client, config.host)
    api = CompManHandler(adapter, config=config)
    return api.handlers[TABLES]
