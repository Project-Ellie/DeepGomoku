import pygame
import pygame_gui as gui

from aegomoku.interfaces import Move
from cmclient.api.game_context import GameContext
from cmclient.gui.emitter import BoardEventEmitter


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

    def __init__(self, board_size: int, context: GameContext):
        self.board_size = board_size
        self.context = context
        width = GRID_SIZE * (self.board_size + 1) + 2 * SIDE_BUFFER + CONTROL_PANE
        height = GRID_SIZE * (self.board_size + 1) + 2 * SIDE_BUFFER
        self.width, self.height = width, height
        pygame.init()
        if self.context.polling_listener is not None:
            pygame.time.set_timer(POLL_NOW, TIME_DELAY)
        self.screen = pygame.display.set_mode((width, height))
        self.clock = pygame.time.Clock()


    def show(self, registered: str, oppenent: str):

        self.redraw()
        pygame.display.set_caption(f"{registered} vs {oppenent}")
        pygame.display.update()

        self.loop()

        return "Done."


    def redraw(self, stones=None):
        # self.screen.fill(COLOR_BOARD)
        # self.draw_grid()
        # self.draw_field_names()
        # current_color = 0
        # if stones is not None:
        #     current_color = self.draw_stones(stones)
        self.draw_controls()

    def draw_controls(self):
        background = pygame.Surface((self.width, self.height))
        background.fill(pygame.Color('#000000'))
        manager = gui.UIManager((self.width, self.height))
        button_rect = pygame.Rect((self.width - 50, 100), (self.width - 100, 50))
        button_rect = pygame.Rect((350, 275), (100, 50))
        self.hello_button = gui.elements.UIButton(relative_rect=button_rect,
                                                  text='Say Hello',
                                                  manager=manager)
        manager.update(.05)
        self.screen.blit(self.screen, (0, 0))


    def draw_field_names(self):
        for i in range(1, self.board_size + 1):

            wh = GRID_SIZE * (self.board_size + 1) + 2 * SIDE_BUFFER

            char = chr(64 + i)
            self.draw_text(char, GRID_SIZE * i + SIDE_BUFFER, wh - GRID_SIZE // 2,
                           COLOR_WHITE, 16)
            self.draw_text(char, GRID_SIZE * i + SIDE_BUFFER, GRID_SIZE // 2,
                           COLOR_WHITE, 16)

            char = str(self.board_size + 1 - i)
            self.draw_text(char, wh-GRID_SIZE//2, GRID_SIZE * i + SIDE_BUFFER,
                           COLOR_WHITE, 16)
            self.draw_text(char, GRID_SIZE//2, GRID_SIZE * i + SIDE_BUFFER,
                           COLOR_WHITE, 16)


    def draw_grid(self):

        middle = self.board_size // 2 + 1

        for i in range(1, self.board_size + 1):
            pygame.draw.line(self.screen, COLOR_WHITE,
                             [GRID_SIZE * i + SIDE_BUFFER, GRID_SIZE + SIDE_BUFFER],
                             [GRID_SIZE * i + SIDE_BUFFER, self.board_size * GRID_SIZE + SIDE_BUFFER], 2)
            pygame.draw.line(self.screen, COLOR_WHITE,
                             [GRID_SIZE + SIDE_BUFFER, GRID_SIZE * i + SIDE_BUFFER],
                             [self.board_size * GRID_SIZE + SIDE_BUFFER, GRID_SIZE * i + SIDE_BUFFER], 2)

        pygame.draw.circle(self.screen, COLOR_WHITE,
                           [GRID_SIZE * middle + SIDE_BUFFER,
                            GRID_SIZE * middle + SIDE_BUFFER], 8)


    def draw_text(self, text, x_pos, y_pos, font_color, font_size):
        ff = pygame.font.Font(pygame.font.get_default_font(), font_size)
        surface, rect = self.text_objects(text, ff, font_color)
        rect.center = (x_pos, y_pos)
        self.screen.blit(surface, rect)

    @staticmethod
    def text_objects(text, font, font_color):
        surface = font.render(text, True, font_color)
        return surface, surface.get_rect()


    def draw_stones(self, stones):
        color = 0
        seqno = 1
        for stone in stones:
            if stone is not None:  # there may be 'non-moves'
                self.draw_stone(stone, color, seqno)
                seqno += 1
                color = 1 - color
        return color


    def draw_stone(self, stone: Move, color, seqno):
        bx, by = stone.c, stone.r
        x = bx * GRID_SIZE + PADDING
        y = by * GRID_SIZE + PADDING
        pygame.draw.circle(self.screen, STONE_COLORS[color],
                           (x, y), GRID_SIZE // 2 - 1)
        self.draw_text(str(seqno), x, y, STONE_COLORS[1-color], 16)


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


    def loop(self):

        emitter = BoardEventEmitter(PADDING, self.board_size, GRID_SIZE, [POLL_NOW])
        ongoing = True

        while ongoing:
            time_delta = self.clock.tick(60)/1000.0
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    ongoing = False

                event = self.move_from_event(event)
                if isinstance(event, Move):
                    current_stones = self.context.move(event)
                    if len(current_stones) > 0:  # move may be illegal
                        self.redraw(current_stones)
                        pygame.display.update()

                        # Now is the AI's turn
                        if self.context.ai is not None:
                            pygame.event.post(pygame.event.Event(AI_NEXT))
                            result = self.context.ai_move()
                            if self.context.winner is not None:
                                print(f"Player {self.context.winner} wins")

                            self.redraw(result)
                            self.manager.draw_ui(self.screen)
                            pygame.display.update()
                else:
                    # self.manager.process_events(event)
                    if event.type == pygame.QUIT:
                        ongoing = False

            # If there's some external system that could have a word in the game
                if event == POLL_NOW:
                    current_stones = self.context.poll()
                    if len(current_stones) > 0:  # move may be illegal
                        self.redraw(current_stones)
                        pygame.display.update()

        return "Done playing."
