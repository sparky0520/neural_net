import numpy as np
import random
from mnist_loader import get_data, one_hot_encode
from network import NeuralNetwork
import os


def train():
    archive = r"c:\Users\sparky\OneDrive\Desktop\neural_net\archive"
    print("Loading data...")
    X_train_raw, y_train_raw, X_test_raw, y_test_raw = get_data(archive)

    # Preprocess training data
    # We need inputs as (784, 1) and labels as (10, 1)
    training_inputs = [np.reshape(x, (784, 1)) for x in X_train_raw]
    training_labels = [np.reshape(y, (10, 1)) for y in one_hot_encode(y_train_raw)]
    training_data = list(zip(training_inputs, training_labels))

    # Preprocess test data
    test_inputs = [np.reshape(x, (784, 1)) for x in X_test_raw]
    test_data = list(zip(test_inputs, y_test_raw))

    # Initialize Network: 784 inputs, 100 hidden neurons, 10 outputs
    net = NeuralNetwork([784, 100, 10])

    epochs = 10
    mini_batch_size = 32
    eta = 0.5  # Learning rate

    print(f"Starting training for {epochs} epochs...")
    for j in range(epochs):
        random.shuffle(training_data)
        mini_batches = [
            training_data[k : k + mini_batch_size]
            for k in range(0, len(training_data), mini_batch_size)
        ]

        for mini_batch in mini_batches:
            net.update_mini_batch(mini_batch, eta)

        # Evaluate performance on test data
        test_correct = 0
        for x, y in test_data:
            pred = net.prediction(x)
            if np.argmax(pred) == y:
                test_correct += 1
        test_accuracy = (test_correct / len(test_data)) * 100

        # Evaluate performance on training data (subset for speed)
        train_correct = 0
        num_train_samples = 2000  # Only check sub-set for speed
        for x, y in training_data[:num_train_samples]:
            pred = net.prediction(x)
            if np.argmax(pred) == np.argmax(y):
                train_correct += 1
        train_accuracy = (train_correct / num_train_samples) * 100

        print(
            f"Epoch {j + 1}: Train Acc: {train_accuracy:.2f}% | Test Acc: {test_accuracy:.2f}%"
        )

    print("\nTraining complete! Saving model...")
    net.save("mnist_model.json")
    print("Model saved as 'mnist_model.json'.")


if __name__ == "__main__":
    train()
