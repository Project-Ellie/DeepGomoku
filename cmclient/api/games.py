import json

from requests import HTTPError

from cmclient.api.basics import CompManConfig, AbstractCompManApi, string_to_stones
from cmclient.api.rest import RestAdapter
from cmclient.gui import board


class GameHandler:
    games_path = "/games/"

    def __init__(self, adapter: RestAdapter, config: CompManConfig,
                 superapi: AbstractCompManApi):
        self.adapter = adapter
        self.config = config
        self.superapi = superapi
        self.current_state = ""

    def handle(self, args):

        if args.play:
            ret = self.play()
            return ret
        return "Nothing to do."

    def play(self):
        player_id = self.config.player
        player = self.superapi.get_player(player_id)
        player_name = player['name']
        print(f'Playing as {player_name}')
        opponent = None
        if self.config.table is not None:
            table = self.superapi.retrieve_table(self.config.table)
            board_state = table['state']
            opponent = self._find_opponent(player_id, table)
            opponent = self.superapi.get_player(opponent)
            opponent = opponent['name']

        ret = board.show(registered=player_name, oppenent=opponent, board_size=self.config.board_size,
                         move_listener=lambda move: self.move(*move),
                         polling_listener=lambda: self.update())
        return ret

    def update(self):
        table = self.superapi.retrieve_table(self.config.table)
        if table['prev_state'] == self.current_state:
            self.current_state = table['state']
            return string_to_stones(table['last_move'])[0]
        else:
            return None

    def move(self, bx, by):
        player_id = self.config.player
        table_id = self.config.table

        body = {'player': player_id, 'table': table_id, 'x': bx, 'y': by}
        try:
            the_table = self.adapter.post(self.games_path + "move/", body=body)
            the_table = json.loads(the_table)
            self.current_state = the_table['state']
        except HTTPError as e:
            print(f"HTTP Error: {e}")
            return None

        return the_table['last_move']

    @staticmethod
    def _find_opponent(player, table):
        if table['first_player'] == player:
            return table['second_player']
        else:
            return table['first_player']
