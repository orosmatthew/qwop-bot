import numpy as np
from util import sigmoid


class NeuralNetwork:

    def __init__(self, input_nodes: int = 24, hidden_nodes: int = 12, output_nodes: int = 4):
        self.input_nodes: int = input_nodes
        self.hidden_nodes: int = hidden_nodes
        self.output_nodes: int = output_nodes

        # Initialize weights from input layer to hidden layer
        # [
        #   [in1 -> h1, in2 -> h1, in3 -> h1, ...],
        #   [in1 -> h2, in2 -> h2, in3 -> h2, ...],
        #   ...
        # ]
        self.weights_ih: np.ndarray = np.random.randn(self.hidden_nodes, self.input_nodes)

        # Initialize weights from hidden layer to output layer
        # [
        #   [h1 -> out1, h2 -> out1, h3 -> out1, ...],
        #   [h1 -> out2, h2 -> out2, h3 -> out2, ...],
        #   ...
        # ]
        self.weights_ho: np.ndarray = np.random.randn(self.output_nodes, self.hidden_nodes)

    def feedforward(self, inputs: np.ndarray):
        # Calculate outputs for hidden layer, by using dot product
        # [the weights to each specific hidden layer neuron] * [the input neurons],
        # each neuron corresponds to each weight
        hidden_outputs = [sigmoid(np.dot(i, inputs)) for i in self.weights_ih]

        # Calculate outputs for output layer, by multiplying
        # (weights of hidden layer output layer) by (hidden layer outputs) and add some bias
        output = [sigmoid(np.dot(i, hidden_outputs)) for i in self.weights_ho]

        return output
