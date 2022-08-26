import pygame
import pygame_gui

from aegomoku.game_play import GamePlay
from aegomoku.gomoku_board import GomokuBoard
from aegomoku.gomoku_game import ConstantBoardInitializer, GomokuGame
from aegomoku.interfaces import Move
from cmclient.api.game_context import GameContext

COLOR_BOARD = (70, 100, 90)
COLOR_WHITE = (255, 255, 255, 0.1)
COLOR_WHITE_STONES = (200, 255, 255)
COLOR_BLACK_STONES = (0, 20, 20)
STONE_COLORS = [COLOR_BLACK_STONES, COLOR_WHITE_STONES]

GRID_SIZE = 45
SIDE_BUFFER = 30
PADDING = GRID_SIZE + SIDE_BUFFER

TIME_DELAY = 50
POLL_NOW = pygame.USEREVENT + 1
CONTROL_PANE = 200


class Game:

    def __init__(self, board_size: int, game_context: GameContext):

        self.board_size = board_size
        self.context = game_context

        width = GRID_SIZE * (self.board_size + 1) + 2 * SIDE_BUFFER + CONTROL_PANE
        height = GRID_SIZE * (self.board_size + 1) + 2 * SIDE_BUFFER
        self.width, self.height = width, height

        pygame.init()

        pygame.display.set_caption('Quick Start')
        self.window_surface = pygame.display.set_mode((width, height))

        self.manager = pygame_gui.UIManager((width, height))

        self.hello_button = pygame_gui.elements.UIButton(relative_rect=pygame.Rect((width-175, 275), (100, 50)),
                                                         text='New Game', manager=self.manager)

        self.clock = pygame.time.Clock()


    def draw_grid(self, background):

        middle = self.board_size // 2 + 1

        for i in range(1, self.board_size + 1):
            pygame.draw.line(background, COLOR_WHITE,
                             [GRID_SIZE * i + SIDE_BUFFER, GRID_SIZE + SIDE_BUFFER],
                             [GRID_SIZE * i + SIDE_BUFFER, self.board_size * GRID_SIZE + SIDE_BUFFER], 2)
            pygame.draw.line(background, COLOR_WHITE,
                             [GRID_SIZE + SIDE_BUFFER, GRID_SIZE * i + SIDE_BUFFER],
                             [self.board_size * GRID_SIZE + SIDE_BUFFER, GRID_SIZE * i + SIDE_BUFFER], 2)

        pygame.draw.circle(background, COLOR_WHITE,
                           [GRID_SIZE * middle + SIDE_BUFFER,
                            GRID_SIZE * middle + SIDE_BUFFER], 8)


    def draw_field_names(self, background):
        for i in range(1, self.board_size + 1):

            wh = GRID_SIZE * (self.board_size + 1) + 2 * SIDE_BUFFER

            char = chr(64 + i)
            self.draw_text(background, char, GRID_SIZE * i + SIDE_BUFFER, wh - GRID_SIZE // 2,
                           COLOR_WHITE, 16)
            self.draw_text(background, char, GRID_SIZE * i + SIDE_BUFFER, GRID_SIZE // 2,
                           COLOR_WHITE, 16)

            char = str(self.board_size + 1 - i)
            self.draw_text(background, char, wh-GRID_SIZE//2, GRID_SIZE * i + SIDE_BUFFER,
                           COLOR_WHITE, 16)
            self.draw_text(background, char, GRID_SIZE//2, GRID_SIZE * i + SIDE_BUFFER,
                           COLOR_WHITE, 16)

    def draw_text(self, background, text, x_pos, y_pos, font_color, font_size):
        ff = pygame.font.Font(pygame.font.get_default_font(), font_size)
        surface, rect = self.text_objects(text, ff, font_color)
        rect.center = (x_pos, y_pos)
        background.blit(surface, rect)

    @staticmethod
    def text_objects(text, font, font_color):
        surface = font.render(text, True, font_color)
        return surface, surface.get_rect()


    def redraw(self, background, stones):

        self.draw_grid(background)
        self.draw_field_names(background)
        if stones is not None:
            self.draw_stones(background, stones)
        self.window_surface.blit(background, (0, 0))
        return background

    def draw_stones(self, background, stones):
        color = 0
        seqno = 1
        for stone in stones:
            if stone is not None:  # there may be 'non-moves'
                self.draw_stone(background, stone, color, seqno)
                seqno += 1
                color = 1 - color


    def draw_stone(self, background, stone: Move, color, seqno):
        bx, by = stone.c, stone.r
        x = bx * GRID_SIZE + PADDING
        y = by * GRID_SIZE + PADDING
        pygame.draw.circle(background, STONE_COLORS[color],
                           (x, y), GRID_SIZE // 2 - 1)
        self.draw_text(background, str(seqno), x, y, STONE_COLORS[1-color], 16)

    def move_from_event(self, event):
        if event.type == pygame.MOUSEBUTTONUP:
            x, y = event.pos
            x = (x - PADDING + GRID_SIZE // 2) // GRID_SIZE
            y = (y - PADDING + GRID_SIZE // 2) // GRID_SIZE
            if not(self.board_size > x >= 0 and self.board_size > y >= 0):
                return event
            else:
                stone = self.context.board.Stone(y * self.board_size + x)
                return stone
        return event


    def run(self):
        is_running = True

        background = pygame.Surface((self.width, self.height))
        background.fill(pygame.Color(COLOR_BOARD))
        self.redraw(background, None)

        while is_running:
            time_delta = self.clock.tick(60)/1000.0
            for event in pygame.event.get():

                if event.type == pygame.QUIT:
                    is_running = False

                if event.type == pygame_gui.UI_BUTTON_PRESSED:
                    if event.ui_element == self.hello_button:
                        print('Hello World!')

                move = self.move_from_event(event)
                if isinstance(move, Move):
                    current_stones = self.context.move(move)
                    background = pygame.Surface((self.width, self.height))
                    background.fill(pygame.Color(COLOR_BOARD))
                    self.redraw(background, None)
                    self.redraw(background, current_stones)

                    # Now is the AI's turn
                    if self.context.ai is not None:
                        result = self.context.ai_move()
                        if self.context.winner is not None:
                            print(f"Player {self.context.winner} wins")

                        self.redraw(background, result)

                self.manager.process_events(event)

            self.manager.update(time_delta)

            self.window_surface.blit(background, (0, 0))
            self.manager.draw_ui(self.window_surface)

            pygame.display.update()


if __name__ == '__main__':
    game_play = GamePlay([])
    board = GomokuBoard(15)
    cbi = ConstantBoardInitializer("")
    game = GomokuGame(board_size=15, initializer=cbi)
    context = GameContext(game, 15)

    Game(15, context).run()
