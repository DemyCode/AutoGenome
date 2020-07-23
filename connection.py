class Connection:
    def __init__(self, father, son, weight):
        self.father: int = father
        self.son: int = son
        self.weight: float = weight

    def copy(self):
        return Connection(father=self.father, son=self.son, weight=self.weight)
