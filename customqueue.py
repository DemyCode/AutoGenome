class CustomQueue:
    def __init__(self):
        self.elements = []

    def push(self, x):
        self.elements.append(x)

    def pop(self):
        return self.elements.pop(0)

    def isempty(self):
        return len(self.elements) == 0
