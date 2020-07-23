from genome import Genome
from time import sleep
import numpy as np
from time import time


class Fitter:
    def __init__(self, genome=None, children=[], evaluate=None):
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
        newfitter = Fitter(genome=self.genome.mutate(), children=[], evaluate=self.evaluate)
        self.children.append(newfitter)

    def survival(self):
        for child in self.children:
            child.survival()
        fittestgenome = self.genome
        fittestscore = self.score
        found = False
        for child in self.children:
            if child.score < fittestscore:
                fittestgenome = child.genome
                fittestscore = child.score
                found = True
        if found:
            self.genome = fittestgenome
            self.score = fittestscore
            self.children = []

    def fit(self, episode=1000, scorebreak=0.001, timebreak=300):
        start_time = time()
        for i in range(episode):
            print('==============')
            print('episode : ', i)
            print('familiy size : ', self.family_size())
            node_size, conn_size = self.genome.size()
            print('Father Complexity : ', node_size, conn_size)
            print('seconds : ', time() - start_time)
            self.set_score()
            self.survival()
            self.grow()
            print('score : ', self.score)
            if time() - start_time > timebreak:
                return
            if self.score < scorebreak:
                return

    def family_size(self):
        size = 1
        for child in self.children:
            size += child.family_size()
        return size
