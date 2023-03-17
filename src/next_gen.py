import random
import numpy as np
import util
import colorsys
import pyray as rl

from neural_network import NeuralNetwork
from character_simulation import CharacterSimulation


# Make child network of two parents
def make_next_gen_child_nn(nn_1: NeuralNetwork, nn_2: NeuralNetwork) -> NeuralNetwork:
    child_weights_ih: np.ndarray = np.zeros(nn_1.weights_ih.shape)
    child_weights_ho: np.ndarray = np.zeros(nn_1.weights_ho.shape)

    mutation_probability = 0.05  # between 0 and 0.5

    # initialize child's ih weights
    for i in range(child_weights_ih.shape[0]):
        for j in range(child_weights_ih.shape[1]):
            rand = random.random()
            if rand < 0.5:
                child_weights_ih[i][j] = nn_1.weights_ih[i][j]
            elif rand > mutation_probability:
                child_weights_ih[i][j] = nn_2.weights_ih[i][j]
            else:
                child_weights_ih[i][j] = random.gauss(0, 0.01)

    # initialize child's ho weights
    for i in range(child_weights_ho.shape[0]):
        for j in range(child_weights_ho.shape[1]):
            rand = random.random()
            if rand < 0.5:
                child_weights_ho[i][j] = nn_1.weights_ho[i][j]
            elif rand > mutation_probability:
                child_weights_ho[i][j] = nn_2.weights_ho[i][j]
            else:
                child_weights_ho[i][j] = random.gauss(0, 0.01)

    child_network: NeuralNetwork = NeuralNetwork()

    child_network.weights_ih = child_weights_ih
    child_network.weights_ho = child_weights_ho
    child_network.bias_ih = nn_1.bias_ih if random.random() < 0.5 else nn_2.bias_ih
    child_network.bias_ho = nn_1.bias_ho if random.random() < 0.5 else nn_2.bias_ho

    return child_network


# Make next 100 children (next generation)
def make_next_gen(generation_list: list[CharacterSimulation]) -> list[CharacterSimulation]:
    children_list: list[CharacterSimulation] = []

    while len(children_list) < 101:
        # randomly select two parents
        parent1, parent2 = random.sample(generation_list, 2)

        # make child network based on the selected parents
        child_network: NeuralNetwork = make_next_gen_child_nn(parent1.neural_network, parent2.neural_network)

        ground_position = 50, 150
        ground_poly = [
            (-50000, -25),
            (-50000, 25),
            (50000, 25),
            (50000, -25),
        ]

        # make a character, add to children_list
        child: CharacterSimulation = CharacterSimulation(ground_position, ground_poly)
        child.neural_network = child_network

        mixed_color = colorsys.rgb_to_hsv(int((parent1.color.r + parent2.color.r)/2),
                                          int((parent1.color.g + parent2.color.g)/2),
                                          int((parent1.color.b + parent2.color.b)/2))

        child.color = rl.color_from_hsv(mixed_color[0], mixed_color[1], mixed_color[2])
        children_list.append(child)

    return children_list
