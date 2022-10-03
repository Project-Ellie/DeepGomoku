from unittest import TestCase

from aegomoku.gomoku_game import Swap2
from aegomoku.interfaces import BLACK, SWAP2_FIRST_THREE, WHITE, SWAP2_AFTER_THREE, SWAP2_AFTER_FIVE, \
    FIRST_PLAYER, OTHER_PLAYER, GAMESTATE_NORMAL

BOARD_SIZE = 15
PASS = BOARD_SIZE * BOARD_SIZE


class Swap2Tests(TestCase):

    def setUp(self) -> None:
        self.game = Swap2(BOARD_SIZE)
        self.board = self.game.get_initial_board()


    def test_game_state_when_no_player_passes(self):

        b = self.board

        self.assertEqual(BLACK, b.get_current_color())
        self.assertEqual(FIRST_PLAYER, b.get_current_player())
        self.assertEqual(SWAP2_FIRST_THREE, b.get_phase())
        self.assert_may_pass(b, False)

        b.act('A1')

        self.assertEqual(WHITE, b.get_current_color())
        self.assertEqual(FIRST_PLAYER, b.get_current_player())
        self.assertEqual(SWAP2_FIRST_THREE, b.get_phase())
        self.assert_may_pass(b, False)

        b.act('B2')

        self.assertEqual(BLACK, b.get_current_color())
        self.assertEqual(FIRST_PLAYER, b.get_current_player())
        self.assertEqual(SWAP2_FIRST_THREE, b.get_phase())
        self.assert_may_pass(b, False)

        b.act('C3')  # first player

        # ----- PHASE SWITCH

        self.assertEqual(WHITE, b.get_current_color())
        self.assertEqual(OTHER_PLAYER, b.get_current_player())
        self.assertEqual(SWAP2_AFTER_THREE, b.get_phase())
        self.assert_may_pass(b, True)

        b.act('D4')  # other player

        self.assertEqual(BLACK, b.get_current_color())
        self.assertEqual(OTHER_PLAYER, b.get_current_player())
        self.assertEqual(SWAP2_AFTER_THREE, b.get_phase())
        self.assert_may_pass(b, True)

        b.act('E5')  # other player

        # ----- PHASE SWITCH:
        # By putting two more stones on the board, the other player acquires the black stones

        self.assertEqual(WHITE, b.get_current_color())
        self.assertEqual(FIRST_PLAYER, b.get_current_player())
        self.assertEqual(SWAP2_AFTER_FIVE, b.get_phase())
        self.assert_may_pass(b, True)

        b.act('F6')  # first_player

        # ----- PHASE SWITCH:
        # By putting a stone on the board, the first player accepts the white stones

        self.assertEqual(BLACK, b.get_current_color())
        self.assertEqual(OTHER_PLAYER, b.get_current_player())
        self.assertEqual(GAMESTATE_NORMAL, b.get_phase())
        self.assert_may_pass(b, False)


    def test_pass_after_three(self):
        b = self.board
        g = self.game

        self.assertEqual(BLACK, g.get_current_player(b))
        self.assertEqual(SWAP2_FIRST_THREE, b.get_phase())
        self.assert_may_pass(b, False)

        b.act('A1')

        self.assertEqual(BLACK, g.get_current_player(b))
        self.assertEqual(SWAP2_FIRST_THREE, b.get_phase())
        self.assert_may_pass(b, False)

        b.act('B2')

        self.assertEqual(BLACK, g.get_current_player(b))
        self.assertEqual(SWAP2_FIRST_THREE, b.get_phase())
        self.assert_may_pass(b, False)

        b.act('C3')

        self.assertEqual(WHITE, g.get_current_player(b))
        self.assertEqual(SWAP2_AFTER_THREE, b.get_phase())
        self.assert_may_pass(b, True)

        b.act(PASS)  # other player

        self.assertEqual(BLACK, g.get_current_player(b))
        self.assertEqual(GAMESTATE_NORMAL, b.get_phase())
        self.assert_may_pass(b, False)

        b.act('D4')  # first player

        self.assertEqual(WHITE, g.get_current_player(b))
        self.assertEqual(GAMESTATE_NORMAL, b.get_phase())
        self.assert_may_pass(b, False)

    def assert_may_pass(self, b, allowed: bool = True):
        valid = self.game.get_valid_moves(b)
        value = 1 if allowed else 0
        self.assertEqual(value, valid[BOARD_SIZE * BOARD_SIZE])


    def test_pass_after_five(self):

        b = self.board
        self.assertEqual(BLACK, b.get_current_player())
        self.assertEqual(SWAP2_FIRST_THREE, b.get_phase())
        self.assert_may_pass(b, False)

        b.act('A1')

        self.assertEqual(BLACK, b.get_current_player())
        self.assertEqual(SWAP2_FIRST_THREE, b.get_phase())
        self.assert_may_pass(b, False)

        b.act('B2')

        self.assertEqual(BLACK, b.get_current_player())
        self.assertEqual(SWAP2_FIRST_THREE, b.get_phase())
        self.assert_may_pass(b, False)

        b.act('C3')

        self.assertEqual(WHITE, b.get_current_player())
        self.assertEqual(SWAP2_AFTER_THREE, b.get_phase())
        self.assert_may_pass(b, True)

        b.act('D4')

        self.assertEqual(WHITE, b.get_current_player())
        self.assertEqual(SWAP2_AFTER_THREE, b.get_phase())
        self.assert_may_pass(b, True)

        b.act('E5')

        self.assertEqual(BLACK, b.get_current_player())
        self.assertEqual(SWAP2_AFTER_FIVE, b.get_phase())
        self.assert_may_pass(b, True)

        b.act(PASS)

        self.assertEqual(WHITE, b.get_current_player())
        self.assertEqual(GAMESTATE_NORMAL, b.get_phase())
        self.assert_may_pass(b, False)

        b.act('F6')

        self.assertEqual(BLACK, b.get_current_player())
        self.assertEqual(GAMESTATE_NORMAL, b.get_phase())
        self.assert_may_pass(b, False)
