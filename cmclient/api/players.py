import json
from typing import List

from cmclient.api.basics import AbstractHandler, AbstractPlayerApi, CompManConfig
from cmclient.api.rest import RestAdapter


class PlayerApi(AbstractPlayerApi):

    def __init__(self, adapter: RestAdapter, config: CompManConfig):
        self.adapter = adapter
        self.config = config

    players_path = "/players/"

    def register_player(self, name: str) -> dict:
        response = self.adapter.post(path=self.players_path + "register/", body={'name': name})
        content = json.loads(response)
        return content

    def list_all_players(self) -> List:
        response = self.adapter.get(self.players_path)
        players_list = json.loads(response)
        return players_list

    def unregister_all_players(self):
        response = self.adapter.post(path=self.players_path + "clear/", body={})
        return json.loads(response)

    def get_player(self, player_id: str):
        response = self.adapter.get(path=self.players_path + player_id + "/")
        return json.loads(response)


class PlayerHandler(AbstractHandler, PlayerApi):

    def __init__(self, adapter: RestAdapter, config: CompManConfig):
        super().__init__(adapter, config)

    def handle(self, args):
        if args.list:
            output = json.dumps(self.list_all_players(), indent=2)
        elif args.register is not None:
            output = self.register_player(args.register)
            output = output['id']
        elif args.clear:
            output = self.unregister_all_players()['message']
        else:
            output = ""
        return output
