from genome import Genome
from autogenome import Fitter
import numpy as np
import gym
from genome import sigmoid
from scipy.special import softmax
from random import shuffle


def xorevaluate(genome: Genome):
    genome.cleaner()
    table = [([0, 0], [0]),
             ([0, 1], [1]),
             ([1, 0], [1]),
             ([1, 1], [0])]
    score = 0.0
    episodes = 20
    for _ in range(episodes):
        shuffle(table)
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


def squareeval(genome: Genome):
    table = []
    for i in range(100):
        table.append([[i], [i * i]])
    score = 0
    for experience in table:
        x, y = experience[0], experience[1]
        output = genome.forward(x)
        score += abs(output[0] - y[0])
    return score


def experience1():
    # Experience 1
    genome = Genome(input_size=2, output_size=1, nodes=[], connections=[])
    fitter = Fitter(genome=genome, evaluate=xorevaluate)
    genome = fitter.fit(episode=5000, timebreak=300)
    genome.cleaner()
    table = [([0, 0], [0]),  # [([x0, x1], [y0, y1, y2]),
             ([0, 1], [1]),  # ... ]
             ([1, 0], [1]),
             ([1, 1], [0])]
    score = 0
    for _ in range(20):
        shuffle(table)
        for experience in table:
            x, y = experience[0], experience[1]
            output = genome.forward(x)
            score += abs(output[0] - y[0])
            print(x, ' : ', output)
    print('score : ', score)
    print(genome.print_graph())


def experience2():
    # Experience 2
    init_genome = Genome(input_size=1, output_size=1, nodes=[], connections=[])
    fitter = Fitter(genome=init_genome, evaluate=squarerooteval)
    fitter.fit(episode=5000, scorebreak=0.1, timebreak=300, mode='min')
    new_genome = fitter.genome
    new_genome.cleaner()
    table = []
    for i in range(100):
        table.append([[i], [np.sqrt(i)]])
    score = 0
    for experience in table:
        x, y = experience[0], experience[1]
        output = new_genome.forward(x)
        score += abs(output[0] - y[0])
        print(x, ' : ', output)
    print('score : ', score)
    print(new_genome.grapher())


def experience3():
    # Experience 3
    init_genome = Genome(input_size=1, output_size=1, nodes=[], connections=[])
    fitter = Fitter(genome=init_genome, evaluate=squareeval)
    fitter.fit(episode=5000, scorebreak=0.1, timebreak=300, mode='min')
    new_genome = fitter.genome
    new_genome.cleaner()
    table = []
    for i in range(0, 100):
        table.append([[i], [i * i]])
    score = 0
    for experience in table:
        x, y = experience[0], experience[1]
        output = new_genome.forward(x)
        score += abs(output[0] - y[0])
        print(x, ' : ', output)
    print('score : ', score)
    print(new_genome.grapher())


import random


def lunareval(genome: Genome):
    env = gym.make('LunarLander-v2')
    number_of_episode = 20
    total_reward = 0
    for i_episode in range(number_of_episode):
        observation = env.reset()
        episode_reward = 0
        for t in range(100):
            # env.render()
            forward_values = genome.forward(observation)
            softmax_values = softmax(forward_values)
            # action_vector = np.array(np.round(softmax_values))
            action = np.random.choice(np.arange(4), p=softmax_values)
            observation, reward, done, info = env.step(int(action))
            if done:
                break
            episode_reward += reward
        total_reward += episode_reward
    env.close()
    print(total_reward / number_of_episode)
    return total_reward / number_of_episode


def experience4():
    genome = Genome(input_size=8, output_size=4)
    fitter = Fitter(genome, evaluate=lunareval)
    new_genome = fitter.fit(episode=50, timebreak=600, mode='max')
    new_genome.cleaner()
    env = gym.make('LunarLander-v2')
    total_reward = 0
    number_of_episode = 20
    for i_episode in range(number_of_episode):
        observation = env.reset()
        episode_reward = 0
        for t in range(100):
            env.render()
            forward_values = new_genome.forward(observation)
            softmax_values = softmax(forward_values)
            # action_vector = np.array(np.round(softmax_values))
            action = np.random.choice(np.arange(4), p=softmax_values)
            observation, reward, done, info = env.step(int(action))
            if done:
                break
            episode_reward += reward
        total_reward += episode_reward
    env.close()
    print('Testing score is :', total_reward / number_of_episode)


def main():
    experience1()


if __name__ == '__main__':
    main()
