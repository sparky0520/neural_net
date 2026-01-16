import numpy as np
from network import NeuralNetwork
import json


def debug_model(model_path="mnist_model.json"):
    """
    Comprehensive debugging tool for the neural network model.
    """
    print("=" * 60)
    print("NEURAL NETWORK DEBUGGING REPORT")
    print("=" * 60)

    # Load the model
    try:
        net = NeuralNetwork.load(model_path)
        print(f"\n✓ Model loaded successfully from '{model_path}'")
    except Exception as e:
        print(f"\n✗ Error loading model: {e}")
        return

    # 1. Architecture
    print(f"\n{'=' * 60}")
    print("1. ARCHITECTURE")
    print(f"{'=' * 60}")
    print(f"Layer sizes: {net.layer_sizes}")
    print(f"Number of layers: {net.num_layers}")

    # 2. Weight Statistics
    print(f"\n{'=' * 60}")
    print("2. WEIGHT STATISTICS")
    print(f"{'=' * 60}")
    for i, w in enumerate(net.weights):
        print(f"\nLayer {i} -> {i + 1}:")
        print(f"  Shape: {w.shape}")
        print(f"  Mean: {np.mean(w):.6f}")
        print(f"  Std: {np.std(w):.6f}")
        print(f"  Min: {np.min(w):.6f}")
        print(f"  Max: {np.max(w):.6f}")
        print(f"  Abs Mean: {np.mean(np.abs(w)):.6f}")

        # Check for dead weights
        near_zero = np.sum(np.abs(w) < 0.001)
        total = w.size
        print(
            f"  Near-zero weights (<0.001): {near_zero}/{total} ({near_zero / total * 100:.2f}%)"
        )

    # 3. Bias Statistics
    print(f"\n{'=' * 60}")
    print("3. BIAS STATISTICS")
    print(f"{'=' * 60}")
    for i, b in enumerate(net.biases):
        print(f"\nLayer {i + 1}:")
        print(f"  Shape: {b.shape}")
        print(f"  Mean: {np.mean(b):.6f}")
        print(f"  Std: {np.std(b):.6f}")
        print(f"  Min: {np.min(b):.6f}")
        print(f"  Max: {np.max(b):.6f}")

    # 4. Output Layer Analysis
    print(f"\n{'=' * 60}")
    print("4. OUTPUT LAYER BIAS ANALYSIS")
    print(f"{'=' * 60}")
    output_bias = net.biases[-1].flatten()
    print("\nOutput biases (one per digit):")
    for digit in range(10):
        print(f"  Digit {digit}: {output_bias[digit]:.6f}")

    # Check if biased toward a specific digit
    max_bias_digit = np.argmax(output_bias)
    min_bias_digit = np.argmin(output_bias)
    print(
        f"\n⚠ Highest bias: Digit {max_bias_digit} ({output_bias[max_bias_digit]:.6f})"
    )
    print(f"⚠ Lowest bias: Digit {min_bias_digit} ({output_bias[min_bias_digit]:.6f})")

    # 5. Test with random inputs
    print(f"\n{'=' * 60}")
    print("5. PREDICTION DISTRIBUTION TEST")
    print(f"{'=' * 60}")
    print("\nTesting with 1000 random inputs...")

    predictions = []
    for _ in range(1000):
        random_input = np.random.rand(784, 1)
        pred = net.prediction(random_input)
        predictions.append(np.argmax(pred))

    print("\nPrediction distribution:")
    for digit in range(10):
        count = predictions.count(digit)
        bar = "█" * (count // 10)
        print(f"  Digit {digit}: {count:4d} {bar}")

    # Check if stuck on one digit
    most_common = max(set(predictions), key=predictions.count)
    most_common_count = predictions.count(most_common)
    if most_common_count > 500:
        print(f"\n⚠ WARNING: Model is heavily biased toward digit {most_common}")
        print(f"   ({most_common_count}/1000 = {most_common_count / 10:.1f}%)")
        print("   This indicates the model is NOT properly trained!")

    # 6. Test with zero input
    print(f"\n{'=' * 60}")
    print("6. ZERO INPUT TEST")
    print(f"{'=' * 60}")
    zero_input = np.zeros((784, 1))
    zero_pred = net.prediction(zero_input)
    zero_digit = np.argmax(zero_pred)
    print(f"\nPrediction for all-zero input: {zero_digit}")
    print("Output probabilities:")
    for digit in range(10):
        print(f"  Digit {digit}: {zero_pred[digit][0] * 100:.2f}%")

    # 7. Recommendations
    print(f"\n{'=' * 60}")
    print("7. RECOMMENDATIONS")
    print(f"{'=' * 60}")

    avg_weight_magnitude = np.mean([np.mean(np.abs(w)) for w in net.weights])

    if avg_weight_magnitude < 0.05:
        print("\n⚠ CRITICAL: Weights are too small!")
        print("   → Change weight initialization in network.py line 18")
        print("   → Replace: np.random.randn(y, x) * 0.01")
        print(
            "   → With: np.random.randn(y, x) * np.sqrt(2.0 / x)  # He initialization"
        )

    if most_common_count > 500:
        print("\n⚠ CRITICAL: Model is stuck predicting one digit!")
        print("   → Retrain the model from scratch")
        print("   → Use better weight initialization (see above)")
        print("   → Consider adjusting learning rate")

    print("\n" + "=" * 60)
    print("END OF REPORT")
    print("=" * 60)


if __name__ == "__main__":
    debug_model()
