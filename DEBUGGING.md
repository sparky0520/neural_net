# Neural Network Debugging Guide

## Problem: Model Predicts Everything as One Digit (e.g., "8")

### Root Cause
**Poor weight initialization** in `network.py` line 18 was using `* 0.01`, making weights 100x too small.

### What Was Wrong
```python
# ❌ BAD - Weights too small
self.weights = [
    np.random.randn(y, x) * 0.01
    for x, y in zip(layer_sizes[:-1], layer_sizes[1:])
]
```

This caused:
- **Vanishing gradients** - signals die during backpropagation
- **Slow learning** - network can't learn meaningful patterns
- **Stuck in local minimum** - defaults to predicting one class

### Solution Applied
```python
# ✅ GOOD - He initialization
self.weights = [
    np.random.randn(y, x) * np.sqrt(2.0 / x)
    for x, y in zip(layer_sizes[:-1], layer_sizes[1:])
]
```

**He initialization** scales weights based on layer size, providing better gradient flow.

---

## Debugging Workflow

### 1. Run the Debug Script
```bash
uv run debug_model.py
```

This will show you:
- Weight statistics (mean, std, min, max)
- Bias statistics
- Prediction distribution on random inputs
- Whether the model is stuck on one digit

### 2. Check for Common Issues

#### Issue: Model stuck on one digit
**Symptoms:**
- Predicts same digit for all inputs
- Random input test shows >50% predictions for one digit

**Solutions:**
- Fix weight initialization (already done ✓)
- Retrain from scratch
- Check learning rate (should be 0.1 - 3.0 for this architecture)

#### Issue: Weights too small
**Symptoms:**
- Average weight magnitude < 0.05
- Many weights near zero

**Solutions:**
- Use He initialization (already done ✓)
- Or Xavier/Glorot: `np.sqrt(1.0 / x)`

#### Issue: Low accuracy after training
**Symptoms:**
- Test accuracy < 85% after 10 epochs
- Training accuracy not improving

**Solutions:**
- Increase epochs (try 20-30)
- Adjust learning rate
- Increase hidden layer size
- Add more hidden layers

### 3. Monitor Training Progress

Good training should show:
```
Epoch 1: Train Acc: 60-70% | Test Acc: 60-70%
Epoch 2: Train Acc: 75-80% | Test Acc: 75-80%
Epoch 3: Train Acc: 82-87% | Test Acc: 82-87%
...
Epoch 10: Train Acc: 92-95% | Test Acc: 90-93%
```

**Red flags:**
- Accuracy stuck at ~10% (random guessing)
- Accuracy stuck at ~11-15% (predicting one class)
- No improvement after epoch 3
- Training accuracy >> Test accuracy (overfitting)

### 4. Test Individual Predictions

After retraining, test with your images:
```bash
uv run main.py --predict 3.jpg
uv run main.py --predict 4.jpg
```

Expected behavior:
- Different digits get different predictions
- Confidence varies based on image quality
- Network "sees" the digit in the thumbnail

---

## Additional Debugging Tools

### Quick Model Check
```python
from network import NeuralNetwork
import numpy as np

net = NeuralNetwork.load("mnist_model.json")

# Test with random inputs
for i in range(10):
    random_input = np.random.rand(784, 1)
    pred = net.prediction(random_input)
    print(f"Test {i}: Predicted {np.argmax(pred)}")

# Should see variety of predictions, not all the same!
```

### Check Weight Magnitudes
```python
net = NeuralNetwork.load("mnist_model.json")
for i, w in enumerate(net.weights):
    print(f"Layer {i}: mean={np.mean(np.abs(w)):.4f}")

# Layer 0 should be ~0.05-0.15
# Layer 1 should be ~0.05-0.15
```

### Visualize Predictions
```python
from mnist_loader import get_data
import numpy as np

archive = r"c:\Users\sparky\OneDrive\Desktop\neural_net\archive"
_, _, X_test, y_test = get_data(archive)

net = NeuralNetwork.load("mnist_model.json")

# Test first 20 images
for i in range(20):
    x = np.reshape(X_test[i], (784, 1))
    pred = net.prediction(x)
    predicted_digit = np.argmax(pred)
    actual_digit = y_test[i]
    
    status = "✓" if predicted_digit == actual_digit else "✗"
    print(f"{status} Image {i}: Predicted {predicted_digit}, Actual {actual_digit}")
```

---

## Hyperparameter Tuning

If accuracy is still low after fixing initialization:

### Learning Rate (`eta`)
- **Current:** 0.5
- **Try:** 0.1, 0.3, 1.0, 3.0
- **Too high:** Accuracy oscillates, doesn't converge
- **Too low:** Very slow learning

### Hidden Layer Size
- **Current:** 100 neurons
- **Try:** 50, 128, 256
- **Larger:** More capacity, slower training
- **Smaller:** Faster training, may underfit

### Epochs
- **Current:** 10
- **Try:** 20, 30
- **More epochs:** Better accuracy (up to a point)
- **Watch for:** Overfitting (train >> test accuracy)

### Mini-batch Size
- **Current:** 32
- **Try:** 16, 64, 128
- **Smaller:** More updates, noisier gradients
- **Larger:** Fewer updates, smoother gradients

---

## Expected Results After Fix

With proper He initialization, you should see:

1. **Training converges properly**
   - Accuracy improves each epoch
   - Reaches 90%+ test accuracy by epoch 10

2. **Predictions are diverse**
   - Random inputs don't all predict same digit
   - Each digit gets predicted roughly 10% of the time on random data

3. **Real images work**
   - Your hand-drawn digits get correctly classified
   - Confidence varies based on image quality

---

## Next Steps

1. ✅ **Fixed weight initialization** (He init)
2. 🔄 **Retrain model** (currently running)
3. ⏳ **Test predictions** (after training completes)
4. 📊 **Run debug script** to verify model health

If problems persist after retraining, run `debug_model.py` and share the output!
