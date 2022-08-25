import abc
from dataclasses import dataclass

PLAYERS = 'players'
TABLES = 'tables'
GAMES = 'games'
STUDY = 'study'
HISTORY = 'history'
CONFIG = 'config'


@dataclass
class CompManConfig:
    host: str = None
    player: str = None
    table: str = None
    board_size: int = 15


class ValidationException(Exception):

    def __init__(self, message):
        super().__init__(message)


class AbstractHandler(abc.ABC):
    def handle(self, args):
        raise NotImplementedError


class AbstractPlayerApi(abc.ABC):
    def register_player(self, name: str) -> str:
        raise NotImplementedError

    def get_player(self, player_id: str) -> dict:
        raise NotImplementedError

    def list_all_players(self):
        raise NotImplementedError

    def unregister_all_players(self, name: str):
        raise NotImplementedError


class AbstractTableApi(abc.ABC):
    def retrieve_table(self, table_id: str) -> dict:
        raise NotImplementedError


class AbstractCompManApi(AbstractPlayerApi, AbstractTableApi, abc.ABC):
    pass


def string_to_stones(encoded):
    """
    returns an array of pairs for a string-encoded sequence
    e.g. [('A',1), ('M',14)] for 'a1m14'
    """
    if encoded == '':
        return [None]
    x, y = encoded[0].upper(), 0
    stones = []
    for c in encoded[1:]:
        if c.isdigit():
            y = 10 * y + int(c)
        else:
            stones.append((x, y))
            x = c.upper()
            y = 0
    stones.append((x, y))
    return stones


def stones_to_string(stones):
    """
    returns a string-encoded sequence for an array of pairs.
    e.g. 'a1m14' for [('A',1), ('M',14)]
    """
    return "".join([(s[0].lower() if type(s[0]) == str else chr(96 + s[0])) + str(s[1]) for s in stones])
