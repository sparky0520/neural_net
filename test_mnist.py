import numpy as np
from mnist_loader import get_data
from network import NeuralNetwork
import os


def test_saved_model():
    archive = r"c:\Users\sparky\OneDrive\Desktop\neural_net\archive"
    model_path = "mnist_model.json"

    if not os.path.exists(model_path):
        print(
            f"Error: Model file '{model_path}' not found. Please run 'train_mnist.py' first."
        )
        return

    print("Loading data...")
    _, _, X_test_raw, y_test_raw = get_data(archive)

    # Preprocess test data
    test_inputs = [np.reshape(x, (784, 1)) for x in X_test_raw]
    test_data = list(zip(test_inputs, y_test_raw))

    print(f"Loading model from '{model_path}'...")
    net = NeuralNetwork.load(model_path)

    print("Testing...")
    correct = 0
    for x, y in test_data:
        pred = net.prediction(x)
        if np.argmax(pred) == y:
            correct += 1

    accuracy = (correct / len(test_data)) * 100
    print(f"Test Accuracy: {accuracy:.2f}%")


if __name__ == "__main__":
    test_saved_model()
