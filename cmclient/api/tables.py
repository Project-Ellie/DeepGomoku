import json
from typing import List

from cmclient.api.basics import CompManConfig, ValidationException
from cmclient.api.rest import RestAdapter


class TableApi:

    tables_path = "/tables/"

    def __init__(self, adapter: RestAdapter, config: CompManConfig):
        self.adapter = adapter
        self.config = config

    def propose_game(self, player_id, discipline, params: dict) -> dict:
        params = "" if dict is None else json.dumps(params)
        body = {'discipline': discipline,
                'player_id': player_id,
                'params': params}

        response = self.adapter.post(self.tables_path + "propose_game/", body=body)
        content = json.loads(response)
        return content

    def list_all_tables(self) -> List:
        response = self.adapter.get(self.tables_path)
        content = json.loads(response)
        return content

    def retrieve_table(self, table_id) -> dict:
        response = self.adapter.get(self.tables_path + table_id + "/")
        return json.loads(response)

    def clear_all_tables(self) -> dict:
        response = self.adapter.post(self.tables_path + "clear/", body={})
        return json.loads(response)

    def join_table(self, player_id, table_id):
        body = {'player_id': player_id,
                'table_id': table_id}
        response = self.adapter.post(self.tables_path + "join_table/", body=body)
        return json.loads(response)


class TableHandler(TableApi):

    def __init__(self, adapter: RestAdapter, config: CompManConfig):
        super().__init__(adapter, config)

    def player_from_id_or_config(self, args):
        if args.id is not None:
            return args.id
        elif self.config.player is not None:
            return self.config.player
        else:
            raise ValidationException("No player in args nor config")

    def table_from_id_or_config(self, args):
        if args.id is not None:
            return args.id
        elif self.config.table is not None:
            return self.config.table
        else:
            raise ValidationException("No table in args nor config")

    def handle(self, args):
        if args.list:
            response = self.list_all_tables()
            output = json.dumps(response, indent=2)
        elif args.propose:
            player_id = self.player_from_id_or_config(args)

            output = self.propose_game(player_id=player_id,
                                       discipline=args.discipline,
                                       params=args.params)['table_id']
        elif args.get:
            table_id = self.table_from_id_or_config(args)

            output = json.dumps(self.retrieve_table(table_id=table_id),
                                indent=2)

        elif args.clear:
            result = self.clear_all_tables()
            return result['message']

        elif args.join is not None:
            player_id = self.player_from_id_or_config(args)
            result = self.join_table(player_id=player_id, table_id=args.join)
            return result

        else:
            output = "Command not understood."

        return output
