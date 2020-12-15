class Node:
    def __init__(self, value: float = 0.0, role='hidden'):
        self.value = value
        self.role = role

    def copy(self):
        return Node(value=0.0, role=self.role)
