from unittest import TestCase

from aegomoku.gomoku_game import Swap2
from aegomoku.interfaces import BLACK, SWAP2_FIRST_THREE, PASS, WHITE, SWAP2_AFTER_THREE, SWAP2_AFTER_FIVE, \
    SWAP2_PASSED_THREE, SWAP2_DONE, SWAP2_PASSED_FIVE, FIRST_PLAYER, OTHER_PLAYER


class Swap2Tests(TestCase):

    def setUp(self) -> None:
        self.game = Swap2(15)
        self.board = self.game.get_initial_board()


    def test_no_pass(self):

        b = self.board
        g = self.game

        self.assertEqual(BLACK, b.get_current_color())
        self.assertEqual(FIRST_PLAYER, b.get_current_player())
        self.assertEqual(SWAP2_FIRST_THREE, b.get_phase())
        self.assertNotIn(PASS, g.get_valid_moves(b))

        b.act('A1')

        self.assertEqual(WHITE, b.get_current_color())
        self.assertEqual(FIRST_PLAYER, b.get_current_player())
        self.assertEqual(SWAP2_FIRST_THREE, b.get_phase())
        self.assertNotIn(PASS, g.get_valid_moves(b))

        b.act('B2')

        self.assertEqual(BLACK, b.get_current_color())
        self.assertEqual(FIRST_PLAYER, b.get_current_player())
        self.assertEqual(SWAP2_FIRST_THREE, b.get_phase())
        self.assertNotIn(PASS, g.get_valid_moves(b))

        b.act('C3')  # first player

        # ----- PHASE SWITCH

        self.assertEqual(WHITE, b.get_current_color())
        self.assertEqual(OTHER_PLAYER, b.get_current_player())
        self.assertEqual(SWAP2_AFTER_THREE, b.get_phase())
        self.assertIn(PASS, g.get_valid_moves(b))

        b.act('D4')  # other player

        self.assertEqual(BLACK, b.get_current_color())
        self.assertEqual(OTHER_PLAYER, b.get_current_player())
        self.assertEqual(SWAP2_AFTER_THREE, b.get_phase())
        self.assertNotIn(PASS, g.get_valid_moves(b))

        b.act('E5')  # other player

        # ----- PHASE SWITCH:
        # By putting two more stones on the board, the other player acquires the black stones

        self.assertEqual(WHITE, b.get_current_color())
        self.assertEqual(FIRST_PLAYER, b.get_current_player())
        self.assertEqual(SWAP2_AFTER_FIVE, b.get_phase())
        self.assertIn(PASS, g.get_valid_moves(b))

        b.act('F6')  # first_player

        # ----- PHASE SWITCH:
        # By putting a stone on the board, the first player accepts the white stones

        self.assertEqual(BLACK, b.get_current_color())
        self.assertEqual(OTHER_PLAYER, b.get_current_player())
        self.assertEqual(SWAP2_DONE, b.get_phase())
        self.assertNotIn(PASS, g.get_valid_moves(b))


    def test_pass_after_three(self):
        b = self.board
        g = self.game

        self.assertEqual(BLACK, g.get_current_player(b))
        self.assertEqual(SWAP2_FIRST_THREE, b.get_phase())
        self.assertNotIn(PASS, g.get_valid_moves(b))

        b.act('A1')

        self.assertEqual(BLACK, g.get_current_player(b))
        self.assertEqual(SWAP2_FIRST_THREE, b.get_phase())
        self.assertNotIn(PASS, g.get_valid_moves(b))

        b.act('B2')

        self.assertEqual(BLACK, g.get_current_player(b))
        self.assertEqual(SWAP2_FIRST_THREE, b.get_phase())
        self.assertNotIn(PASS, g.get_valid_moves(b))

        b.act('C3')

        self.assertEqual(WHITE, g.get_current_player(b))
        self.assertEqual(SWAP2_AFTER_THREE, b.get_phase())
        self.assertIn(PASS, g.get_valid_moves(b))

        b.act(PASS)  # other player

        self.assertEqual(BLACK, g.get_current_player(b))
        self.assertEqual(SWAP2_PASSED_THREE, b.get_phase())
        self.assertNotIn(PASS, g.get_valid_moves(b))

        b.act('D4')  # first player

        self.assertEqual(WHITE, g.get_current_player(b))
        self.assertEqual(SWAP2_DONE, b.get_phase())
        self.assertNotIn(PASS, g.get_valid_moves(b))


    def test_pass_after_five(self):

        b = self.board
        g = self.game

        self.assertEqual(BLACK, g.get_current_player(b))
        self.assertEqual(SWAP2_FIRST_THREE, b.get_phase())
        self.assertNotIn(PASS, g.get_valid_moves(b))

        b.act('A1')

        self.assertEqual(BLACK, g.get_current_player(b))
        self.assertEqual(SWAP2_FIRST_THREE, b.get_phase())
        self.assertNotIn(PASS, g.get_valid_moves(b))

        b.act('B2')

        self.assertEqual(BLACK, g.get_current_player(b))
        self.assertEqual(SWAP2_FIRST_THREE, b.get_phase())
        self.assertNotIn(PASS, g.get_valid_moves(b))

        b.act('C3')

        self.assertEqual(WHITE, g.get_current_player(b))
        self.assertEqual(SWAP2_AFTER_THREE, b.get_phase())
        self.assertIn(PASS, g.get_valid_moves(b))

        b.act('D4')

        self.assertEqual(WHITE, g.get_current_player(b))
        self.assertEqual(SWAP2_AFTER_THREE, b.get_phase())
        self.assertNotIn(PASS, g.get_valid_moves(b))

        b.act('E5')

        self.assertEqual(BLACK, g.get_current_player(b))
        self.assertEqual(SWAP2_AFTER_FIVE, b.get_phase())
        self.assertIn(PASS, g.get_valid_moves(b))

        b.act(PASS)

        self.assertEqual(WHITE, g.get_current_player(b))
        self.assertEqual(SWAP2_PASSED_FIVE, b.get_phase())
        self.assertNotIn(PASS, g.get_valid_moves(b))

        b.act('F6')

        self.assertEqual(BLACK, g.get_current_player(b))
        self.assertEqual(SWAP2_DONE, b.get_phase())
        self.assertNotIn(PASS, g.get_valid_moves(b))
