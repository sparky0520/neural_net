"""Quick model health check - simplified version"""

import numpy as np
from network import NeuralNetwork

print("Loading model...")
net = NeuralNetwork.load("mnist_model.json")

print("\n" + "=" * 50)
print("QUICK HEALTH CHECK")
print("=" * 50)

# 1. Weight magnitudes
print("\n1. WEIGHT MAGNITUDES:")
for i, w in enumerate(net.weights):
    avg = np.mean(np.abs(w))
    status = "✅" if 0.03 < avg < 0.3 else "❌"
    print(f"   Layer {i}: {avg:.4f} {status}")

# 2. Prediction test
print("\n2. PREDICTION DIVERSITY TEST (100 random inputs):")
preds = []
for _ in range(100):
    x = np.random.rand(784, 1)
    pred = net.prediction(x)
    preds.append(np.argmax(pred))

counts = [preds.count(i) for i in range(10)]
for digit in range(10):
    bar = "█" * (counts[digit] // 2)
    print(f"   Digit {digit}: {counts[digit]:2d} {bar}")

max_count = max(counts)
if max_count > 50:
    print(
        f"\n   ❌ WARNING: Digit {counts.index(max_count)} predicted {max_count}% of time!"
    )
    print("   Model is biased - needs retraining")
else:
    print("\n   ✅ Predictions are diverse - model looks healthy")

print("\n" + "=" * 50)
