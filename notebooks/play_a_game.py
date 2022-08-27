from cmclient.api.basics import CompManConfig
from cmclient.api.study import StudyHandler

config = CompManConfig(board_size=15)

StudyHandler(config).handle()

if __name__ == '__main__':
    StudyHandler(config).handle()
