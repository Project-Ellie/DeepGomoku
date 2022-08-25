import pygame

from aegomoku.interfaces import Move
from cmclient.ai import get_player
from cmclient.gui.emitter import BoardEventEmitter


COLOR_BOARD = (70, 100, 90)
COLOR_WHITE = (255, 255, 255)
COLOR_WHITE_STONES = (200, 255, 255)
COLOR_BLACK_STONES = (0, 20, 20)
STONE_COLORS = [COLOR_BLACK_STONES, COLOR_WHITE_STONES]

GRID_SIZE = 45
SIDE_BUFFER = 30
PADDING = GRID_SIZE + SIDE_BUFFER

TIME_DELAY = 500
POLL_NOW = pygame.USEREVENT + 1


def geometry(board_size: int):
    middle = board_size // 2 + 1
    wh = GRID_SIZE * (board_size + 1) + 2 * SIDE_BUFFER
    physical_board = [wh] * 2
    return middle, wh, physical_board


def show(registered: str, oppenent: str, board_size: int,
         move_listener, polling_listener):

    _, _, physical_board = geometry(board_size)
    pygame.init()
    if polling_listener is not None:
        pygame.time.set_timer(POLL_NOW, TIME_DELAY)
    screen = pygame.display.set_mode(physical_board)

    current_color = redraw(screen, board_size)
    pygame.display.set_caption(f"{registered} vs {oppenent}")
    pygame.display.update()

    loop(move_listener, polling_listener, screen, board_size, current_color)

    return "Done."


def redraw(screen, board_size, stones=None):
    screen.fill(COLOR_BOARD)
    draw_grid(screen, board_size)
    draw_field_names(screen, board_size)
    current_color = 0
    if stones is not None:
        current_color = draw_stones(screen, stones)
    return current_color


def draw_field_names(screen, board_size):
    for i in range(1, board_size + 1):

        _, wh, __ = geometry(board_size)

        char = chr(64 + i)
        draw_text(screen, char,
                  GRID_SIZE * i + SIDE_BUFFER, wh-GRID_SIZE//2, COLOR_WHITE, 16)
        draw_text(screen, char,
                  GRID_SIZE * i + SIDE_BUFFER, GRID_SIZE//2, COLOR_WHITE, 16)

        char = str(board_size + 1 - i)
        draw_text(screen, char,
                  wh-GRID_SIZE//2, GRID_SIZE * i + SIDE_BUFFER, COLOR_WHITE, 16)
        draw_text(screen, char,
                  GRID_SIZE//2, GRID_SIZE * i + SIDE_BUFFER, COLOR_WHITE, 16)


def draw_grid(screen, board_size):

    middle = board_size // 2 + 1

    for i in range(1, board_size + 1):
        pygame.draw.line(screen, COLOR_WHITE,
                         [GRID_SIZE * i + SIDE_BUFFER, GRID_SIZE + SIDE_BUFFER],
                         [GRID_SIZE * i + SIDE_BUFFER, board_size * GRID_SIZE + SIDE_BUFFER], 2)
        pygame.draw.line(screen, COLOR_WHITE,
                         [GRID_SIZE + SIDE_BUFFER, GRID_SIZE * i + SIDE_BUFFER],
                         [board_size * GRID_SIZE + SIDE_BUFFER, GRID_SIZE * i + SIDE_BUFFER], 2)

    pygame.draw.circle(screen, COLOR_WHITE,
                       [GRID_SIZE * middle + SIDE_BUFFER,
                        GRID_SIZE * middle + SIDE_BUFFER], 8)


def draw_text(screen, text, x_pos, y_pos, font_color, font_size):
    ff = pygame.font.Font(pygame.font.get_default_font(), font_size)
    surface, rect = text_objects(text, ff, font_color)
    rect.center = (x_pos, y_pos)
    screen.blit(surface, rect)


def text_objects(text, font, font_color):
    surface = font.render(text, True, font_color)
    return surface, surface.get_rect()


def draw_stones(screen, stones):
    color = 0
    seqno = 1
    for stone in stones:
        if stone is not None:  # there may be 'non-moves'
            draw_stone(screen, stone, color, seqno)
            seqno += 1
            color = 1 - color
    return color


def draw_stone(screen, stone: Move, color, seqno):
    bx, by = stone.c, stone.r
    x = bx * GRID_SIZE + PADDING
    y = by * GRID_SIZE + PADDING
    pygame.draw.circle(screen, STONE_COLORS[color],
                       (x, y), GRID_SIZE // 2 - 1)
    draw_text(screen, str(seqno), x, y, STONE_COLORS[1-color], 16)


def loop(move_listener, polling_listener, screen, board_size, current_color):

    ai = get_player(board_size)

    emitter = BoardEventEmitter(PADDING, board_size, GRID_SIZE, [POLL_NOW])
    ongoing = True
    seqno = 1
    while ongoing:
        event = emitter.get()
        if event == "EXIT":
            ongoing = False
        elif event == POLL_NOW:
            current_stones = polling_listener()
            if len(current_stones) > 0:  # move may be illegal
                redraw(screen, board_size, current_stones)
                draw_stones(screen, current_stones)
                pygame.display.update()
                seqno += 1
                current_color = 1 - current_color
        elif isinstance(event, Move):
            board = move_listener(event)
            current_stones = board.get_stones()
            if len(current_stones) > 0:  # move may be illegal
                redraw(screen, board_size, current_stones)
                pygame.display.update()
                seqno += 1

                # Now is the AI's turn
                board, _ = ai.move(board)
                current_stones = board.get_stones()
                redraw(screen, board_size, current_stones)
                pygame.display.update()
        else:
            print(f"Unknown event {event}. Ignoring...")

    return "Done playing."
