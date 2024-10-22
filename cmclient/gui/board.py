import pygame
import pygame_gui

from aegomoku.interfaces import Move
from cmclient.api.game_context import GameContext


COLOR_BOARD = (70, 100, 90)
COLOR_BRIGHT_WHITE = (255, 255, 255)
COLOR_WHITE = (192, 192, 192)
COLOR_BLACK = (64, 64, 64)
COLOR_GRAY = (136, 136, 136)
COLOR_RED = (255, 0, 0)

# for the seqno of the stones
COLOR_WHITE_STONES = (160, 192, 192)
COLOR_BLACK_STONES = (92, 64, 64)
STONE_COLORS = [COLOR_BLACK_STONES, COLOR_WHITE_STONES]

GRID_SIZE = 55
SIDE_BUFFER = 30
PADDING = GRID_SIZE + SIDE_BUFFER

TIME_DELAY = 50
CONTROL_PANE = 200

AI_NEXT = pygame.USEREVENT + 3


class UI:

    def __init__(self, game_context: GameContext, base_path):

        self.board_size = game_context.game.board_size
        self.context = game_context

        width = GRID_SIZE * (self.board_size + 1) + 2 * SIDE_BUFFER + CONTROL_PANE
        height = GRID_SIZE * (self.board_size + 1) + 2 * SIDE_BUFFER
        self.width, self.height = width, height
        self.disp_threshold = .003
        self.show_advice = "Policy"
        self.base_path = base_path

        self.clock = pygame.time.Clock()

        self.window_surface = None
        self.manager = None
        self.new_game_button = None
        self.pass_button = None
        self.advice_button = None
        pygame.mixer.init()
        self.stone_on_board = pygame.mixer.Sound(self.base_path + "piece.wav")
        self.stone_on_board.set_volume(.1)
        self.ponder = False

    def show(self, title):
        pygame.init()

        pygame.display.set_caption(title)
        self.window_surface = pygame.display.set_mode((self.width, self.height))

        self.manager = pygame_gui.UIManager((self.width, self.height))

        self.new_game_button = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect((self.width - 125, 50), (100, 50)),
            text='New Game', manager=self.manager)

        self.pass_button = pygame_gui.elements.UIButton(relative_rect=pygame.Rect((self.width-125, 125), (100, 50)),
                                                        text='Pass', manager=self.manager)

        rect = pygame.Rect((self.width - 125, 200), (100, 50))
        self.advice_button = pygame_gui.elements.UIDropDownMenu(relative_rect=rect,
                                                                options_list=['Policy', 'MCTS', "None"],
                                                                starting_option='Policy',
                                                                manager=self.manager)

        rect = pygame.Rect((self.width - 125, 425), (100, 50))
        self.advice_button = pygame_gui.elements.UIDropDownMenu(relative_rect=rect,
                                                                options_list=["Don't", "Ponder"],
                                                                starting_option="Don't",
                                                                manager=self.manager)

        rect = pygame.Rect((self.width - 125, 500), (100, 50))
        self.advice_button = pygame_gui.elements.UIDropDownMenu(relative_rect=rect,
                                                                options_list=[str(i) for i in range(400, 2200, 200)],
                                                                starting_option="400",
                                                                manager=self.manager)

        self.run()


    def draw_grid(self, background, grid_color=COLOR_GRAY):

        middle = self.board_size // 2 + 1

        for i in range(1, self.board_size + 1):
            pygame.draw.line(background, grid_color,
                             [GRID_SIZE * i + SIDE_BUFFER, GRID_SIZE + SIDE_BUFFER],
                             [GRID_SIZE * i + SIDE_BUFFER, self.board_size * GRID_SIZE + SIDE_BUFFER], 2)
            pygame.draw.line(background, grid_color,
                             [GRID_SIZE + SIDE_BUFFER, GRID_SIZE * i + SIDE_BUFFER],
                             [self.board_size * GRID_SIZE + SIDE_BUFFER, GRID_SIZE * i + SIDE_BUFFER], 2)

        pygame.draw.circle(background, grid_color,
                           [GRID_SIZE * middle + SIDE_BUFFER,
                            GRID_SIZE * middle + SIDE_BUFFER], 8)


    def draw_field_names(self, background, color=COLOR_GRAY):

        wh = GRID_SIZE * (self.board_size + 1) + 2 * SIDE_BUFFER

        for i in range(1, self.board_size + 1):

            char = chr(64 + i)
            self.draw_text(background, char, GRID_SIZE * i + SIDE_BUFFER, wh - GRID_SIZE // 2,
                           color, 16)
            self.draw_text(background, char, GRID_SIZE * i + SIDE_BUFFER, GRID_SIZE // 2,
                           color, 16)

            char = str(self.board_size + 1 - i)
            self.draw_text(background, char, wh-GRID_SIZE//2, GRID_SIZE * i + SIDE_BUFFER,
                           color, 16)
            self.draw_text(background, char, GRID_SIZE//2, GRID_SIZE * i + SIDE_BUFFER,
                           color, 16)

        aegomoku = pygame.image.load(self.base_path + "aegomoku1.png").convert_alpha()
        aegomoku = pygame.transform.smoothscale(aegomoku, (144, 36))
        background.blit(aegomoku, (wh-GRID_SIZE//2 + 60,
                                   GRID_SIZE * self.board_size + SIDE_BUFFER + 35))


    def draw_text(self, background, text, x_pos, y_pos, font_color, font_size, bg=None):
        ff = pygame.font.Font(pygame.font.get_default_font(), font_size)
        surface, rect = self.text_objects(text, ff, font_color)
        if bg:
            pygame.draw.rect(surface, bg, rect)

        rect.center = (x_pos, y_pos)
        background.blit(surface, rect)

    @staticmethod
    def text_objects(text, font, font_color):
        surface = font.render(text, True, font_color)
        return surface, surface.get_rect()


    def redraw(self, stones):

        background = pygame.Surface((self.width, self.height))
        texture = pygame.image.load(self.base_path + "wooden_board.jpg").convert()
        texture = pygame.transform.smoothscale(texture, (self.width, self.height))
        background.blit(texture, (0, 0))

        self.draw_grid(background)
        self.draw_field_names(background)
        if stones is not None and len(stones) > 0:
            self.draw_stones(background, stones)
        self.window_surface.blit(background, (0, 0))
        self.draw_advice(background)
        self.advice_button.update(.1)
        return background

    def draw_stones(self, background, stones):
        white = pygame.image.load(self.base_path + "white.png").convert_alpha()
        white = pygame.transform.smoothscale(white, (48, 48))
        black = pygame.image.load(self.base_path + "black.png").convert_alpha()
        black = pygame.transform.smoothscale(black, (48, 48))
        images = [black, white]

        color = 0
        seqno = 1
        for stone in stones[:-1]:
            if stone is not None:  # there may be 'non-moves'
                self.draw_stone(background, stone, color, images[color], seqno)
                seqno += 1
                color = 1 - color
        self.draw_stone(background, stones[-1], color, images[color], seqno, mark=True)


    def draw_stone(self, background, stone: Move, color, image, seqno, mark=False):
        bx, by = stone.c, stone.r
        x = bx * GRID_SIZE + PADDING
        y = by * GRID_SIZE + PADDING

        background.blit(image, (x-25, y-25))

        self.draw_text(background, str(seqno), x, y, STONE_COLORS[1-color], 16)
        if mark:
            s = GRID_SIZE // 2
            rect = pygame.Rect((x-s, y-s), (2 * s, 2 * s))
            pygame.draw.rect(background, COLOR_RED, rect=rect, width=2)

    def draw_advice(self, background):
        (p_advice, p_value), (m_advice, m_value) = self.context.get_advice()

        pygame.draw.rect(background, COLOR_WHITE, ((self.width - 125, 275), (100, 50)))
        self.draw_text(background, str(round(float(p_value), 5)), self.width - 75, 300, COLOR_BLACK, 16)

        pygame.draw.rect(background, COLOR_WHITE, ((self.width - 125, 350), (100, 50)))
        self.draw_text(background, str(round(float(m_value), 5)), self.width - 75, 375, COLOR_BLACK, 16)

        if self.show_advice == 'Policy':
            advice = p_advice
        elif self.show_advice == 'MCTS':
            advice = m_advice
        else:
            return

        for i, prob in enumerate(advice):
            if prob > self.disp_threshold:
                r, c = divmod(i, self.board_size)
                pos = self.context.board.Stone(r, c)
                if pos not in self.context.board.get_stones():
                    x = c * GRID_SIZE + PADDING
                    y = r * GRID_SIZE + PADDING

                    intensity = min(1, 2 * prob)
                    color = (255 * intensity, (1 - intensity) * 255, 128 * intensity)

                    pygame.draw.circle(background, color,
                                       (x, y), GRID_SIZE // 5)


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

        stones = self.context.board.get_stones()
        new_image = self.redraw(stones)

        while is_running:
            time_delta = self.clock.tick(5)/1000.0
            for event in pygame.event.get():

                if event.type == pygame.QUIT:
                    is_running = False

                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_LEFT:
                        stones = self.context.bwd()
                        new_image = self.redraw(stones)
                    if event.key == pygame.K_RIGHT:
                        stones = self.context.fwd()
                        new_image = self.redraw(stones)

                elif event.type == pygame_gui.UI_BUTTON_PRESSED:
                    if event.ui_element == self.new_game_button:
                        initial_stones = self.context.new_game()
                        new_image = self.redraw(initial_stones)
                    elif event.ui_element == self.pass_button:
                        self.context.ai_active = True
                        pygame.event.post(pygame.event.Event(AI_NEXT))

                elif event.type == pygame_gui.UI_SELECTION_LIST_NEW_SELECTION:
                    if event.text in ['MCTS', 'Policy', 'None']:
                        self.show_advice = event.text
                        new_image = self.redraw(self.context.board.get_stones())
                    elif event.text == 'Ponder':
                        self.ponder = True
                    elif event.text == "Don't":
                        self.ponder = False
                    elif event.text in [str(i) for i in range(400, 2200, 200)]:
                        self.context.num_simu = int(event.text)

                elif event.type == AI_NEXT:

                    current_stones = self.context.ai_move()
                    new_image = self.redraw(current_stones)
                    self.stone_on_board.play()

                    if self.context.winner is not None:
                        print(f"Player {self.context.winner} wins")

                move = self.move_from_event(event)
                if isinstance(move, Move):
                    current_stones = self.context.move(move)
                    self.stone_on_board.play()
                    if current_stones is None:  # rogue move, ignoring
                        continue
                    new_image = self.redraw(current_stones)

                    # Now is the AI's turn
                    if self.context.ai_active:
                        pygame.event.post(pygame.event.Event(AI_NEXT))

                self.manager.process_events(event)

                if self.ponder:
                    self.context.ponder()
                # end for
            # end outer

            self.manager.update(time_delta)

            self.window_surface.blit(new_image, (0, 0))
            self.manager.draw_ui(self.window_surface)

            pygame.display.update()
