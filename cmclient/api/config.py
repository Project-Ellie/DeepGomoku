import json

from cmclient.api.rest import RestAdapter
from compman.apps import CompManConfig


class ConfigHandler:

    def __init__(self, adapter: RestAdapter, config: CompManConfig):
        self.adapter = adapter
        self.config = config

    def handle(self, args):
        if args.list:
            output = json.dumps(self.config.__dict__, indent=2)
            return output
