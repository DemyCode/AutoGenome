class Node:
    def __init__(self, visited=False, value: float = 0.0, role='hidden', node_id=0):
        self.visited = visited
        self.value = value
        self.role = role
        self.node_id = node_id

    def copy(self):
        return Node(value=0.0, role=self.role, node_id=self.node_id, visited=False)
