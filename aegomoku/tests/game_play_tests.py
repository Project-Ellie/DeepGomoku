from unittest import TestCase

from aegomoku.game_play import GamePlay


class GamePlayTest(TestCase):

    def setUp(self) -> None:
        self.initial_stones = [1, 2, 3, 4, 5, 6, 7, 8]
        self.game = GamePlay(self.initial_stones)

    def test_initial_play(self):

        self.assertTrue(self.game.root.move == 1)
        self.assertTrue(self.game.current.move == 8)

        self.assertEqual(self.initial_stones, self.game.get_stones())

    def test_bwd(self):
        self.game.bwd()
        self.assertTrue(self.game.current.move == 7)
        for _ in range(10):
            self.game.bwd()
        self.assertIsNone(self.game.current)

    def test_fwd(self):
        self.game.bwd()
        self.game.fwd()
        self.assertTrue(self.game.current.move == 8)
        self.game.fwd()
        self.assertTrue(self.game.current.move == 8)

    def test_new_branch(self):
        for _ in range(4):
            self.game.bwd()
        self.assertTrue(self.game.current.move == 4)
        self.game.fwd(9)
        self.assertTrue(self.game.current.move == 9)
        self.game.bwd()
        self.assertTrue(self.game.current.move == 4)
        self.assertRaises(ValueError, self.game.fwd)
        self.game.fwd(5)
        self.assertTrue(self.game.current.move == 5)
        self.game.bwd()
        self.game.fwd(9)
        self.assertTrue(self.game.current.move == 9)
        self.game.fwd(10)
        self.assertTrue(self.game.current.move == 10)
        self.game.bwd()
        self.game.bwd()
        next_moves = [n.move for n in self.game.next_nodes()]
        self.assertTrue(next_moves == [5, 9])

    def test_cut_branch(self):
        for _ in range(3):
            self.game.bwd()
        self.game.cut()
        self.assertTrue(self.game.current.move == 4)
