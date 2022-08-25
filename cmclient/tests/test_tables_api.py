from rest_framework.test import APITestCase

from cmclient.tests.tools import get_player_handler, get_table_handler


class TestTableAPI(APITestCase):

    def test_propose_game_api(self):
        player_handler = get_player_handler(self.client)
        first_player_id = player_handler.register_player("Wolfie")['id']
        joining_player_id = player_handler.register_player("Harry")['id']

        table_handler = get_table_handler(self.client)
        params = {"size": 15}
        table_id = table_handler.propose_game(player_id=first_player_id,
                                              discipline="gomoku", params=params)['table_id']
        all_tables = table_handler.list_all_tables()
        self.assertEquals(all_tables[0]['id'], table_id)

        table_handler.join_table(table_id=table_id, player_id=joining_player_id)

        table = table_handler.retrieve_table(table_id)

        self.assertEquals(table['first_player'], table['current_player'])
        self.assertEquals(table['first_player'], str(first_player_id))
        self.assertEquals(table['second_player'], str(joining_player_id))

        table_handler.clear_all_tables()
        tables = table_handler.list_all_tables()
        self.assertEquals(0, len(tables))
