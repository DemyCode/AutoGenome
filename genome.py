from connection import Connection
from node import Node
from random import randrange, random, uniform
import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def relu(x):
    return x if x > 0 else 0


class Genome:
    def __init__(self, input_size=0, output_size=0, nodes=None, connections=None):
        self.node_id = 0
        if connections is None:
            connections = []
        if nodes is None:
            nodes = []
        self.input_size = input_size
        self.output_size = output_size
        self.nodes: [Node] = nodes
        self.connections: [Connection] = connections
        for i in range(input_size):
            nodeinput = Node(visited=False, value=0.0, role='input', node_id=self.node_id)
            self.node_id += 1
            self.nodes.append(nodeinput)
        for i in range(output_size):
            nodeoutput = Node(visited=False, value=0.0, role='output', node_id=self.node_id)
            self.node_id += 1
            self.nodes.append(nodeoutput)
        for i in range(input_size):
            for j in range(output_size):
                connection = Connection(i, input_size + j, uniform(-0.01, 0.01))
                self.connections.append(connection)

    def forward(self, input_layer=None):
        # Initialization
        if len(input_layer) != self.input_size:
            raise Exception('input and genome expected input are not the same size')
        for x in self.nodes:
            x.visited = False
        queue = []

        input_nodes = [x for x in self.nodes if x.role == 'input']
        for i in range(self.input_size):
            input_nodes[i].value = input_layer[i]
            input_nodes[i].visited = True
            queue += [x.son for x in self.connections if x.father == input_nodes[i].node_id]

        queue = np.unique(np.array(queue)).tolist()
        queue = [self.find_node(node_id) for node_id in queue]

        # Selection
        while len(queue) != 0:
            cur_node: Node = queue.pop()
            cur_node.visited = True
            # print(cur_node)
            fathers = [(self.find_node(x.father), x.weight) for x in self.connections if x.son == cur_node.node_id]

            # Calculation Node Value
            cur_node.value = 0.0
            for father, weight in fathers:
                cur_node.value += weight * father.value
            if cur_node.role == 'hidden':
                cur_node.value = sigmoid(cur_node.value)

            # Setting path to follow
            sons = [self.find_node(x.son) for x in self.connections if x.father is cur_node.node_id]
            for son in sons:
                if not son.visited:
                    queue.append(son)

        # Return Values
        return [x.value for x in self.nodes if x.role == 'output']

    def mutate(self, deepness):
        newgen = self.copy()
        initial_probability = 0.01

        if random() < initial_probability * deepness:
            randnodea = newgen.nodes[randrange(0, len(newgen.nodes))]
            randnodeb = newgen.nodes[randrange(0, len(newgen.nodes))]
            newgen.connections.append(Connection(father=randnodea.node_id, son=randnodeb.node_id,
                                                 weight=uniform(-initial_probability * deepness, initial_probability * deepness)))

        if random() < initial_probability * deepness and len(newgen.connections) != 0:
            newnode = Node(visited=False, value=0.0, role='hidden', node_id=newgen.node_id)
            newgen.nodes.append(newnode)
            newgen.node_id += 1
            randconnec = newgen.connections[randrange(0, len(newgen.connections))]
            firstconn = Connection(father=randconnec.father, son=newnode.node_id, weight=1.0)
            seccon = Connection(father=newnode.node_id, son=randconnec.son, weight=randconnec.weight)
            newgen.connections.append(firstconn)
            newgen.connections.append(seccon)

        for x in newgen.connections:
            x.weight += uniform(-initial_probability, initial_probability)

        return newgen

    def copy(self):
        newgen = Genome()
        newgen.node_id = self.node_id
        newgen.input_size = self.input_size
        newgen.output_size = self.output_size
        newgen.nodes = []
        newgen.connections = []
        for x in self.nodes:
            newgen.nodes.append(x.copy())
        for x in self.connections:
            newgen.connections.append(x.copy())
        return newgen

    def find_node(self, node_id):
        for x in self.nodes:
            if node_id == x.node_id:
                return x
        return None

    def grapher(self):
        res = "digraph G { "
        for node in self.nodes:
            res += str(node.node_id) + ";"
        for connection in self.connections:
            res += str(connection.father) + " -> " + str(connection.son) + " [label=" + str(connection.weight) + "];"
        res += "}"
        return res

    def size(self):
        return len(self.nodes), len(self.connections)

    def cleaner(self):
        for node in self.nodes:
            node.value = 0.0
