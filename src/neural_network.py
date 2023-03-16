import numpy as np

from util import sigmoid


class NeuralNetwork:

    # TODO: Make the number of hidden layers variable
    def __init__(self, input_nodes: int = 24, hidden_nodes: int = 12, output_nodes: int = 4):
        self.bias_ih: float = np.random.randn()
        self.bias_ho: float = np.random.randn()

        # Initialize weights from input layer to hidden layer
        # [
        #   [in1 -> h1, in2 -> h1, in3 -> h1, ...],
        #   [in1 -> h2, in2 -> h2, in3 -> h2, ...],
        #   ...
        # ]
        self.weights_ih: np.ndarray = np.random.randn(hidden_nodes, input_nodes + 1)  # + 1 is for bias

        # Initialize weights from hidden layer to output layer
        # [
        #   [h1 -> out1, h2 -> out1, h3 -> out1, ...],
        #   [h1 -> out2, h2 -> out2, h3 -> out2, ...],
        #   ...
        # ]
        self.weights_ho: np.ndarray = np.random.randn(output_nodes, hidden_nodes + 1)  # + 1 is for bias

    def feedforward(self, inputs: np.ndarray) -> np.ndarray:
        # Calculate outputs for hidden layer, by using dot product
        # [the weights to each specific hidden layer neuron] * [the input neurons],
        # each neuron corresponds to each weight
        inputs_with_bias = np.append(inputs, self.bias_ih)
        hidden_outputs = np.asarray([sigmoid(np.dot(i, inputs_with_bias)) for i in self.weights_ih])

        # Calculate outputs for output layer, by multiplying
        # (weights of hidden layer output layer) by (hidden layer outputs) and add some bias
        hidden_with_bias = np.append(hidden_outputs, self.bias_ho)
        output = np.asarray([sigmoid(np.dot(i, hidden_with_bias)) for i in self.weights_ho])

        return output

    def load_data(self, data: dict) -> None:
        self.bias_ih = data["bias_ih"]
        self.bias_ho = data["bias_ho"]
        self.weights_ih = np.asarray(data["weights_ih"])
        self.weights_ho = np.asarray(data["weights_ho"])

    def output_data(self) -> dict:
        data = {
            "bias_ih": self.bias_ih,
            "bias_ho": self.bias_ho,
            "weights_ih": self.weights_ih.tolist(),
            "weights_ho": self.weights_ho.tolist()
        }
        return data
