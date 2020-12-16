from connection import Connection
from node import Node
from random import randrange, random, uniform, randint
import numpy as np
from typing import List


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def relu(x):
    return x if x > 0 else 0


def leaky_relu(x):
    return x if x > 0 else x * 0.01


def identity(x):
    return x


class Genome:
    def __init__(self, input_size=0, output_size=0, nodes: List[Node] = None, connections: List[Connection] = None):
        if connections is None:
            connections = []
        if nodes is None:
            nodes = []
        self.input_size: int = input_size
        self.output_size: int = output_size
        self.nodes: List[Node] = nodes
        self.connections: List[Connection] = connections
        for i in range(input_size):
            nodeinput = Node(value=0.0, role='input')
            self.nodes.append(nodeinput)
        for i in range(output_size):
            nodeoutput = Node(value=0.0, role='output')
            self.nodes.append(nodeoutput)
        for i in [i for i, x in enumerate(self.nodes) if x.role == 'input']:
            for j in [j for j, x in enumerate(self.nodes) if x.role == 'output']:
                self.connections.append(Connection(i, j, 0.0))

    def forward(self, input_layer=None):
        # Initialization
        if len(input_layer) != self.input_size:
            raise Exception('input and genome expected input are not the same size')
        # Preparing Queue
        enqueued = [False] * len(self.nodes)
        queue = []
        # Enqueuing input node and setting initial values
        for (i, node) in [(i, node) for i, node in enumerate(self.nodes) if node.role == 'input']:
            node.value = input_layer[i]
            enqueued[i] = True
            queue.append(i)

        # Selection
        while len(queue) != 0:
            cur_id: int = queue.pop(0)
            cur_node = self.nodes[cur_id]
            # Calculation Node Value
            if cur_node.role != 'input':
                fathers = [(conn_id, (self.nodes[conn.left_id], conn.weight)) for (conn_id, conn) in
                           enumerate(self.connections) if conn.right_id == cur_id]
                cur_node.value = 0.0
                for conn_id, (father, weight) in fathers:
                    cur_node.value += weight * father.value
                if cur_node.role == 'hidden':
                    cur_node.value = leaky_relu(cur_node.value)

            # Setting path to follow
            sons_id = [conn.right_id for conn in self.connections if conn.left_id == cur_id]
            for son_id in sons_id:
                if not enqueued[son_id]:
                    queue.append(son_id)
                    enqueued[son_id] = True
        # self.connections = [conn for conn, marker in zip(self.connections, used_connections) if marker]
        # Return Values
        return [x.value for x in self.nodes if x.role == 'output']

    def expect_value(self, cur_id, trace):
        if trace[cur_id] or self.nodes[cur_id].role == 'input':
            return self.nodes[cur_id].value
        trace[cur_id] = True
        result = 0.0
        fathers_id_weights = [(conn.left_id, conn.weight) for conn in self.connections if conn.right_id == cur_id]
        for father_id, weight in fathers_id_weights:
            result += weight * self.expect_value(father_id, trace)
        result = leaky_relu(result) if self.nodes[cur_id].role != 'output' else result
        return result

    def newforward(self, input_layer):
        # Initialization
        if len(input_layer) != self.input_size:
            raise Exception('input and genome expected input are not the same size')
        for i, node in [(i, node) for i, node in enumerate(self.nodes) if node.role == 'input']:
            node.value = input_layer[i]
        for i, node in [(i, node) for i, node in enumerate(self.nodes) if node.role == 'output']:
            trace = [False] * len(self.nodes)
            node.value = self.expect_value(i, trace)
        return [x.value for x in self.nodes if x.role == 'output']


    def mutate(self):
        newgen = self.copy()
        GD = 1 / 100

        def pognegGD() -> float:
            modifier = [2*(-GD), -GD, 0.0, GD, 2*GD]
            return modifier[randrange(0, len(modifier))]

        # pognegGD = lambda: GD if randint(0, 1) == 0 else -GD

        # mutate
        # if random() < initial_probability:
        #     leftnodeid = randrange(0, len(newgen.nodes))
        #     rightnodeid = randrange(0, len(newgen.nodes))
        #     newgen.connections.append(Connection(left_id=leftnodeid, right_id=rightnodeid,
        #                                          weight=pognegGD()))
        #
        # if random() < initial_probability and len(newgen.connections) != 0:
        #     randcon = newgen.connections.pop(randrange(0, len(newgen.connections)))
        #
        #     newnode = Node()
        #     newgen.nodes.append(newnode)
        #     newnode_id = len(newgen.nodes) - 1
        #
        #     firstconn = Connection(left_id=randcon.left_id, right_id=newnode_id, weight=1.0)
        #     seccon = Connection(left_id=newnode_id, right_id=randcon.right_id, weight=randcon.weight)
        #     newgen.connections.append(firstconn)
        #     newgen.connections.append(seccon)

        # if len(newgen.connections) != 0:
        #     newgen.connections[randrange(0, len(newgen.connections))].weight += pognegGD()

        # Best Version
        for conn in newgen.connections:
            conn.weight += pognegGD()
            # if random() < GD:
            #     conn.weight = pognegGD()

        return newgen

    def copy(self):
        newgen = Genome()
        newgen.input_size = self.input_size
        newgen.output_size = self.output_size
        newgen.nodes = []
        newgen.connections = []
        for node in self.nodes:
            newgen.nodes.append(node.copy())
        for conn in self.connections:
            newgen.connections.append(conn.copy())
        return newgen

    def print_graph(self):
        res = ''
        res += 'digraph G {'
        res += 'rankdir="LR"'
        input_nodes = [(i, x) for i, x in enumerate(self.nodes) if x.role == 'input']
        res += 'subgraph cluster_0 { style=filled; color=lightgrey; label="inputs";'
        for i, node in input_nodes:
            res += '{} [label="{}:{}"]'.format(i, i, node.role)
        res += '}'
        for i, node in enumerate(self.nodes):
            res += '{} [label="{}:{}"]'.format(i, i, node.role)
        for conn in self.connections:
            res += '{} -> {} [label="{}"]'.format(conn.left_id, conn.right_id, conn.weight)
        output_nodes = [(i, x) for i, x in enumerate(self.nodes) if x.role == 'output']
        res += 'subgraph cluster_1 { style=filled; color=lightgrey; label="outputs";'
        for i, node in output_nodes:
            res += '{} [label="{}:{}"]'.format(i, i, node.role)
        res += '}'
        return res + '}'

    def conn_size(self):
        return len(self.connections)

    def node_size(self):
        return len(self.nodes)

    def size(self):
        return self.conn_size() + self.node_size()

    def cleaner(self):
        for node in self.nodes:
            node.value = 0.0
