from rest_framework.test import APITestCase

from cmclient.tests.tools import get_player_handler


class PlayerTest(APITestCase):

    def test_successful_player_registration(self):
        handler = get_player_handler(self.client)
        new_id = handler.register_player(name='Willie')['id']
        player = handler.list_all_players()[0]
        self.assertEqual(player['name'], 'Willie')
        self.assertEqual(player['id'], new_id)

    def test_empty_player_list(self):
        handler = get_player_handler(self.client)
        all_players = handler.list_all_players()
        self.assertEquals(0, len(all_players))

    def test_successful_player_list(self):
        handler = get_player_handler(self.client)
        all_players = handler.list_all_players()
        self.assertEqual(0, len(all_players))

        handler.register_player(name='Willie')
        handler.register_player(name='Billie')
        all_players = handler.list_all_players()

        self.assertEqual(2, len(all_players))
