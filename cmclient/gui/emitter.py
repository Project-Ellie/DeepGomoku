import pygame

from aegomoku.gomoku_board import GomokuBoard


class BoardEventEmitter:
    def __init__(self, padding, board_size, grid_size, user_events=None):
        self.user_events = user_events if user_events is not None else []
        self.padding = padding
        self.board_size = board_size
        self.grid_size = grid_size
        self.Stone = GomokuBoard(board_size).Stone

    def get(self):
        while True:
            event = pygame.event.wait()
            if event.type == pygame.MOUSEBUTTONUP:
                x, y = event.pos
                x = (x - self.padding + self.grid_size // 2) // self.grid_size
                y = (y - self.padding + self.grid_size // 2) // self.grid_size
                if not(self.board_size > x >= 0 and self.board_size > y >= 0):
                    continue

                stone = self.Stone(y * self.board_size + x)
                return stone

            elif event.type == pygame.KEYDOWN and event.key == 27:
                return "EXIT"
            elif event.type == pygame.WINDOWCLOSE:
                return "EXIT"
            elif event.type in self.user_events:
                return event.type
