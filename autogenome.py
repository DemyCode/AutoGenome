from genome import Genome
from time import sleep
import numpy as np
from time import time


class Fitter:
    def __init__(self, genome=None, children=None, evaluate=None):
        if children is None:
            children = []
        self.genome: Genome = genome
        self.children = children
        self.score = None
        self.evaluate = evaluate

    def set_score(self):
        if self.score is None:
            self.score = self.evaluate(self.genome)
        for child in self.children:
            child.set_score()

    def grow(self):
        for child in self.children:
            child.grow()
        number_of_child = len(self.children)
        newfitter = Fitter(genome=self.genome.mutate(number_of_child), children=None,
                           evaluate=self.evaluate)
        self.children.append(newfitter)

    def survival(self, mode):
        for child in self.children:
            child.survival(mode)
        fittestgenome = self.genome
        fittestscore = self.score
        found = False
        for child in self.children:
            if (mode == 'min' and child.score < fittestscore) or (mode == 'max' and child.score > fittestscore):
                fittestgenome = child.genome
                fittestscore = child.score
                found = True

        if found:
            self.genome = fittestgenome
            self.score = fittestscore
            self.children = []

    def fit(self, episode=1000, scorebreak=None, timebreak=300, mode='min'):
        start_time = time()
        for i in range(episode):
            # if (i % 100 == 0):
            print('==============')
            print('episode : ', i)
            print('familiy size : ', self.family_size())
            node_size, conn_size = self.genome.size()
            # if (i % 100 == 0):
            print('Father nodes : ', node_size, 'Father Connections : ', conn_size, 'Father Childs : ', len(self.children))
            print('seconds : ', time() - start_time)
            self.set_score()
            self.survival(mode)
            self.grow()
            # if (i % 100 == 0):
            print('score : ', self.score)
            if time() - start_time > timebreak:
                return self.genome
            if scorebreak is not None and self.score < scorebreak:
                return self.genome
        return self.genome

    def family_size(self):
        size = 1
        for child in self.children:
            size += child.family_size()
        return size
