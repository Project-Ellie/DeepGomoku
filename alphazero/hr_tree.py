from alphazero.interfaces import Board


class TreeNode:

    def __init__(self, parent, key: str, action, hrs, value, level=1):
        self.parent = parent
        self.level = level
        self.key = key
        self.action = action
        self.hrs = hrs
        self.value = value
        self.children = {}

    def add_child(self, action, board: Board, value):
        key = board.get_string_representation()
        child = TreeNode(self, key, board.Stone(action), board.get_stones(), value, level=self.level + 1)  # noqa
        self.children[action] = child
        return child

    def __str__(self):
        tab = self.level * "  "
        return f"{self.hrs}: {self.value}\n{tab}" + f"\n{tab}".join([str(c) for c in self.children.values()])

    __repr__ = __str__
