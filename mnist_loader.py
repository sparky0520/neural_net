import numpy as np
import struct
import os

def load_mnist_images(filename):
    with open(filename, 'rb') as f:
        # Read the header
        magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
        if magic != 2051:
            raise ValueError(f"Magic number mismatch, expected 2051, got {magic}")
        # Read the image data
        images = np.frombuffer(f.read(), dtype=np.uint8).reshape(num, rows * cols)
    return images / 255.0  # Normalize to [0, 1]

def load_mnist_labels(filename):
    with open(filename, 'rb') as f:
        # Read the header
        magic, num = struct.unpack(">II", f.read(8))
        if magic != 2049:
            raise ValueError(f"Magic number mismatch, expected 2049, got {magic}")
        # Read the label data
        labels = np.frombuffer(f.read(), dtype=np.uint8)
    return labels

def one_hot_encode(labels, num_classes=10):
    one_hot = np.zeros((labels.size, num_classes))
    one_hot[np.arange(labels.size), labels] = 1
    return one_hot

def get_data(archive_path):
    train_images_path = os.path.join(archive_path, 'train-images.idx3-ubyte')
    train_labels_path = os.path.join(archive_path, 'train-labels.idx1-ubyte')
    test_images_path = os.path.join(archive_path, 't10k-images.idx3-ubyte')
    test_labels_path = os.path.join(archive_path, 't10k-labels.idx1-ubyte')

    train_X = load_mnist_images(train_images_path)
    train_y = load_mnist_labels(train_labels_path)
    test_X = load_mnist_images(test_images_path)
    test_y = load_mnist_labels(test_labels_path)

    return train_X, train_y, test_X, test_y

if __name__ == "__main__":
    # Test loading
    archive = r"c:\Users\sparky\OneDrive\Desktop\neural_net\archive"
    X_train, y_train, X_test, y_test = get_data(archive)
    print(f"Loaded {len(X_train)} training images and {len(X_test)} test images.")
    print(f"Image shape: {X_train[0].shape}")
    print(f"First label: {y_train[0]}")
