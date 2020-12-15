class Connection:
    def __init__(self, left_id, right_id, weight):
        self.left_id: int = left_id
        self.right_id: int = right_id
        self.weight: float = weight

    def copy(self):
        return Connection(left_id=self.left_id, right_id=self.right_id, weight=self.weight)
