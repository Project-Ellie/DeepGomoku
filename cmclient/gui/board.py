import pygame
import pygame_gui

from aegomoku.interfaces import Move
from cmclient.api.game_context import GameContext


COLOR_BOARD = (70, 100, 90)
COLOR_WHITE = (255, 255, 255)
COLOR_WHITE_STONES = (200, 255, 255)
COLOR_BLACK_STONES = (0, 20, 20)
STONE_COLORS = [COLOR_BLACK_STONES, COLOR_WHITE_STONES]

GRID_SIZE = 45
SIDE_BUFFER = 30
PADDING = GRID_SIZE + SIDE_BUFFER

TIME_DELAY = 50
CONTROL_PANE = 200

POLL_NOW = pygame.USEREVENT + 1
AI_NEXT = pygame.USEREVENT + 2


class UI:

    def __init__(self, game_context: GameContext):

        self.board_size = game_context.game.board_size
        self.context = game_context

        width = GRID_SIZE * (self.board_size + 1) + 2 * SIDE_BUFFER + CONTROL_PANE
        height = GRID_SIZE * (self.board_size + 1) + 2 * SIDE_BUFFER
        self.width, self.height = width, height

    def show(self, title):
        pygame.init()

        pygame.display.set_caption(title)
        self.window_surface = pygame.display.set_mode((self.width, self.height))

        self.manager = pygame_gui.UIManager((self.width, self.height))

        self.hello_button = pygame_gui.elements.UIButton(relative_rect=pygame.Rect((self.width-175, 175), (100, 50)),
                                                         text='New Game', manager=self.manager)

        self.pass_button = pygame_gui.elements.UIButton(relative_rect=pygame.Rect((self.width-175, 275), (100, 50)),
                                                        text='Pass', manager=self.manager)

        self.clock = pygame.time.Clock()

        self.run()


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


    def redraw(self, stones):

        background = pygame.Surface((self.width, self.height))
        background.fill(pygame.Color(COLOR_BOARD))
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

        new_image = self.redraw(None)

        while is_running:
            time_delta = self.clock.tick(5)/1000.0
            for event in pygame.event.get():

                if event.type == pygame.QUIT:
                    is_running = False
                    continue

                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_LEFT:
                        stones = self.context.bwd()
                        new_image = self.redraw(stones)

                if event.type == pygame_gui.UI_BUTTON_PRESSED:
                    if event.ui_element == self.hello_button:
                        initial_stones = self.context.new_game()
                        new_image = self.redraw(initial_stones)
                    elif event.ui_element == self.pass_button:
                        self.context.ai_active = True
                        continue

                elif event.type == AI_NEXT:
                    current_stones = self.context.ai_move()
                    new_image = self.redraw(current_stones)

                    if self.context.winner is not None:
                        print(f"Player {self.context.winner} wins")

                move = self.move_from_event(event)
                if isinstance(move, Move):
                    current_stones = self.context.move(move)
                    if current_stones is None:  # rogue move, ignoring
                        continue
                    new_image = self.redraw(current_stones)

                    # Now is the AI's turn
                    if self.context.ai_active:
                        pygame.event.post(pygame.event.Event(AI_NEXT))

                self.manager.process_events(event)
                # end for
            # end outer

            self.manager.update(time_delta)

            self.window_surface.blit(new_image, (0, 0))
            self.manager.draw_ui(self.window_surface)

            pygame.display.update()
