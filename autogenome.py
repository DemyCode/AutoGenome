from genome import Genome
from time import sleep
import numpy as np
from time import time
from typing import List


class Fitter:
    def __init__(self, genome=None, children=None, evaluate=None):
        if children is None:
            children = []
        self.genome: Genome = genome
        self.children: List[Fitter] = children
        self.score = None
        self.evaluate = evaluate
        self.history = {'familysizes': [],
                        'genomesizes': [],
                        'nodesizes': [],
                        'connsizes': [],
                        'scores': [],
                        'episodetimes': [],
                        'episodes': 0}

    def set_score(self):
        if self.score is None:
            self.score = self.evaluate(self.genome)
        for child in self.children:
            child.set_score()

    def grow(self):
        for child in self.children:
            child.grow()
        explorer = Fitter(genome=self.genome.mutate(), evaluate=self.evaluate)
        self.children.append(explorer)

    def survival(self, mode):
        for child in self.children:
            child.survival(mode)
        found = False
        for child in self.children:
            if (mode == 'min' and child.score < self.score) or (mode == 'max' and child.score > self.score):
                self.genome = child.genome
                self.score = child.score
                found = True
            # elif self.score == child.score and child.genome.size() < self.genome.size():
            #     self.genome = child.genome
            # self.score = child.score
            # found = True
        if found:
            self.children = []

    def fit(self, episode=1000, scorebreak=None, timebreak=None, mode='min'):
        start_time = time()
        for i in range(episode):
            self.set_score()
            self.survival(mode)
            self.grow()

            # history
            self.history['familysizes'].append(self.family_size())
            self.history['genomesizes'].append(self.genome.size())
            self.history['nodesizes'].append(self.genome.node_size())
            self.history['connsizes'].append(self.genome.conn_size())
            self.history['scores'].append(self.score)
            self.history['episodetimes'].append(time() - start_time)

            print('==============')
            print('episode : ', i)
            print('familiy size : ', self.family_size())
            node_size, conn_size = self.genome.node_size(), self.genome.conn_size()
            print('Genome odes : ', node_size, 'Genome Connections : ', conn_size)
            print('seconds : ', time() - start_time)
            print('score : ', self.score)
            print(self.genome.print_graph())
            if timebreak is not None and time() - start_time > timebreak:
                break
            if scorebreak is not None:
                if mode == 'max' and self.score > scorebreak:
                    break
                if mode == 'min' and self.score < scorebreak:
                    break
        return self.genome

    def family_size(self):
        size = 1
        for child in self.children:
            size += child.family_size()
        return size
