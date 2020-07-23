from genome import Genome
from autogenome import Fitter
from random import randrange
import math
import numpy as np


def xorevaluate(genome: Genome):
    table = [[[0, 0], [0]],
             [[0, 1], [1]],
             [[1, 0], [1]],
             [[1, 1], [0]]]
    score = 0.0
    for experience in table:
        x, y = experience[0], experience[1]
        output = genome.forward(x)
        score += abs(output[0] - y[0])
    return score


def squarerooteval(genome: Genome):
    score = 0.0
    table = []
    for i in range(100):
        table.append([[i], [np.sqrt(i)]])
    for experience in table:
        x, y = experience[0], experience[1]
        output = genome.forward(x)
        score += abs(output[0] - y[0])
    return score


def experience1():
    # Experience 1
    init_genome = Genome(input_size=2, output_size=1, nodes=[], connections=[])
    fitter = Fitter(genome=init_genome, evaluate=xorevaluate)
    fitter.fit(episode=10000, scorebreak=0.001)
    new_genome = fitter.genome
    table = [[[0, 0], [0]],  # [[[x0, x1], [y0, y1, y2]],
             [[0, 1], [1]],  # ... ]
             [[1, 0], [1]],
             [[1, 1], [0]]]
    for experience in table:
        x, y = experience[0], experience[1]
        output = new_genome.forward(x)
        print(x, ' : ', output)
    print(new_genome.grapher())


def experience2():
    # Experience 1
    init_genome = Genome(input_size=1, output_size=1, nodes=[], connections=[])
    fitter = Fitter(genome=init_genome, evaluate=squarerooteval)
    fitter.fit(episode=10000, scorebreak=0.01)
    new_genome = fitter.genome
    table = []
    for i in range(100):
        table.append([[i], [np.sqrt(i)]])
    for experience in table:
        x, y = experience[0], experience[1]
        output = new_genome.forward(x)
        print(x, ' : ', output)
    print(new_genome.grapher())


def main():
    experience2()


if __name__ == '__main__':
    main()
