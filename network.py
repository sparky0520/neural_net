import numpy as np
import json


class NeuralNetwork:
    def __init__(self, layer_sizes):
        """
        Initialize the neural network with given layer sizes.
        layer_sizes: list of integers, e.g., [784, 128, 10]
        """
        self.layer_sizes = layer_sizes
        self.num_layers = len(layer_sizes)

        # Initialize weights and biases
        # Using He initialization for better gradient flow
        #
        # For each layer connection:
        #   x = number of neurons in the INPUT layer (fan-in)
        #   y = number of neurons in the OUTPUT layer (fan-out)
        #   Weight matrix shape: (y, x) - each row connects to one output neuron
        #
        # He init: multiply by sqrt(2 / x) where x is the input size
        # This scaling prevents vanishing/exploding gradients
        #
        # Example for [784, 100, 10] network:
        #   Layer 0->1: x=784, y=100, weights shape (100, 784), scale=sqrt(2/784)≈0.050
        #   Layer 1->2: x=100, y=10,  weights shape (10, 100),  scale=sqrt(2/100)≈0.141
        self.weights = [
            np.random.randn(y, x) * np.sqrt(2.0 / x)
            for x, y in zip(layer_sizes[:-1], layer_sizes[1:])
        ]
        self.biases = [np.zeros((y, 1)) for y in layer_sizes[1:]]

    def sigmoid(self, z):
        """Sigmoid activation function."""
        return 1.0 / (1.0 + np.exp(-z))

    def sigmoid_derivative(self, z):
        """Derivative of the sigmoid activation function."""
        return self.sigmoid(z) * (1 - self.sigmoid(z))

    def softmax(self, z):
        """Softmax activation function for the output layer."""
        exp_z = np.exp(z - np.max(z))  # Numerical stability
        return exp_z / exp_z.sum(axis=0, keepdims=True)

    def prediction(self, a):
        """
        Forward propagation.
        'a' is the input vector (784, 1).
        Returns the output of the network (10, 1).
        """
        for i in range(len(self.weights)):
            w = self.weights[i]
            b = self.biases[i]
            z = np.dot(w, a) + b
            if i == len(self.weights) - 1:
                a = self.softmax(z)
            else:
                a = self.sigmoid(z)
        return a

    def feedback(self, x, y):
        """
        Backward propagation (Stochastic Gradient Descent).
        x: input vector (784, 1)
        y: target label (one-hot encoded) (10, 1)
        Returns the gradients (nabla_b, nabla_w) for the cost function.
        """
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        # --- Forward Pass ---
        activation = x
        activations = [x]  # list to store all activations, layer by layer
        zs = []  # list to store all z vectors, layer by layer

        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            z = np.dot(w, activation) + b
            zs.append(z)
            if i == len(self.weights) - 1:
                activation = self.softmax(z)
            else:
                activation = self.sigmoid(z)
            activations.append(activation)

        # --- Backward Pass ---

        # Output layer error (using Cross-Entropy loss derivative with Softmax)
        # delta = derivative of cost w.r.t z of output layer
        # For Categorical Cross-Entropy + Softmax, delta is simply (predictions - targets)
        delta = activations[-1] - y
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())

        # Backpropagate through hidden layers
        for layer_idx in range(2, self.num_layers):
            z = zs[-layer_idx]
            sp = self.sigmoid_derivative(z)
            delta = np.dot(self.weights[-layer_idx + 1].transpose(), delta) * sp
            nabla_b[-layer_idx] = delta
            nabla_w[-layer_idx] = np.dot(delta, activations[-layer_idx - 1].transpose())

        return (nabla_b, nabla_w)

    def update_mini_batch(self, mini_batch, eta):
        """
        Update weights and biases using gradient descent with a mini-batch.
        mini_batch: list of (x, y) tuples
        eta: learning rate
        """
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.feedback(x, y)
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]

        self.weights = [
            w - (eta / len(mini_batch)) * nw for w, nw in zip(self.weights, nabla_w)
        ]
        self.biases = [
            b - (eta / len(mini_batch)) * nb for b, nb in zip(self.biases, nabla_b)
        ]

    def save(self, filename):
        """Save the neural network to a file."""
        data = {
            "layer_sizes": self.layer_sizes,
            "weights": [w.tolist() for w in self.weights],
            "biases": [b.tolist() for b in self.biases],
        }
        with open(filename, "w") as f:
            json.dump(data, f)

    @classmethod
    def load(cls, filename):
        """Load a neural network from a file."""
        with open(filename, "r") as f:
            data = json.load(f)
        net = cls(data["layer_sizes"])
        net.weights = [np.array(w) for w in data["weights"]]
        net.biases = [np.array(b) for b in data["biases"]]
        return net
