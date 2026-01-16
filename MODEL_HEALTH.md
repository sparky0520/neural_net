# How to Determine Neural Network Model Health

## Quick Health Check Criteria

### ✅ **HEALTHY MODEL**
1. **Weight Magnitudes**: 0.05 - 0.3 (average absolute value)
2. **Prediction Distribution**: Each digit predicted ~10% of time on random inputs
3. **Test Accuracy**: >85% after 10 epochs
4. **Training Progress**: Steady improvement each epoch
5. **Bias Values**: Relatively balanced across output neurons

### ❌ **UNHEALTHY MODEL**
1. **Weight Magnitudes**: <0.01 (too small) or >1.0 (too large)
2. **Prediction Distribution**: One digit predicted >50% of time
3. **Test Accuracy**: <70% after 10 epochs
4. **Training Progress**: Stuck or oscillating
5. **Bias Values**: One output heavily biased

---

## Detailed Health Metrics

### 1. **Weight Statistics**

#### What to Check:
```python
for i, w in enumerate(net.weights):
    mean_abs = np.mean(np.abs(w))
    print(f"Layer {i}: {mean_abs:.4f}")
```

#### Healthy Ranges:
- **Layer 0 (784→100)**: 0.04 - 0.08
  - Formula: `sqrt(2/784) ≈ 0.050`
- **Layer 1 (100→10)**: 0.12 - 0.18
  - Formula: `sqrt(2/100) ≈ 0.141`

#### Red Flags:
- ❌ **Too small (<0.01)**: Vanishing gradients, can't learn
- ❌ **Too large (>0.5)**: Exploding gradients, unstable
- ❌ **Many zeros**: Dead neurons, poor initialization

#### Why It Matters:
Weights control signal strength through the network. Too small = signals die out. Too large = signals explode.

---

### 2. **Prediction Distribution Test**

#### What to Check:
```python
predictions = []
for _ in range(1000):
    random_input = np.random.rand(784, 1)
    pred = net.prediction(random_input)
    predictions.append(np.argmax(pred))

# Count each digit
for digit in range(10):
    count = predictions.count(digit)
    print(f"Digit {digit}: {count}/1000")
```

#### Healthy Pattern:
```
Digit 0: 80-120  (8-12%)
Digit 1: 80-120  (8-12%)
Digit 2: 80-120  (8-12%)
...
Digit 9: 80-120  (8-12%)
```
Each digit should be predicted roughly equally on random noise.

#### Red Flags:
- ❌ **One digit >500**: Model is stuck/biased
- ❌ **One digit 0**: Dead output neuron
- ❌ **Extreme imbalance**: Poor training

#### Why It Matters:
A healthy model shouldn't have strong preferences on random data. If it predicts "8" for everything, it's not actually learning patterns - it's just biased.

---

### 3. **Training Accuracy Progression**

#### What to Check:
Look at the output during training:
```
Epoch 1: Train Acc: 65.50% | Test Acc: 64.23%
Epoch 2: Train Acc: 78.25% | Test Acc: 77.89%
Epoch 3: Train Acc: 84.10% | Test Acc: 83.45%
...
```

#### Healthy Pattern:
| Epoch | Train Acc | Test Acc | Status |
|-------|-----------|----------|--------|
| 1     | 60-70%    | 60-70%   | ✅ Good start |
| 3     | 80-85%    | 80-85%   | ✅ Learning well |
| 5     | 88-92%    | 86-90%   | ✅ Converging |
| 10    | 92-96%    | 90-93%   | ✅ Well-trained |

#### Red Flags:
- ❌ **Stuck at ~10%**: Random guessing (broken model)
- ❌ **Stuck at ~11-15%**: Predicting one class only
- ❌ **No improvement after epoch 3**: Learning rate too low or weights too small
- ❌ **Train 95%, Test 70%**: Overfitting
- ❌ **Oscillating**: Learning rate too high

#### Why It Matters:
Training curves tell you if the model is actually learning. Steady improvement = healthy learning.

---

### 4. **Bias Analysis**

#### What to Check:
```python
output_bias = net.biases[-1].flatten()
for digit in range(10):
    print(f"Digit {digit}: {output_bias[digit]:.4f}")
```

#### Healthy Pattern:
```
Digit 0: -0.2 to +0.2
Digit 1: -0.2 to +0.2
Digit 2: -0.2 to +0.2
...
```
Biases should be relatively balanced.

#### Red Flags:
- ❌ **One bias >>others** (e.g., digit 8: +5.0): Model heavily biased
- ❌ **One bias <<others** (e.g., digit 3: -5.0): That digit never predicted
- ❌ **All biases very large** (>10): Potential training instability

#### Why It Matters:
Output biases act as "default preferences". Large biases mean the model has strong preferences before even looking at the input.

---

### 5. **Zero Input Test**

#### What to Check:
```python
zero_input = np.zeros((784, 1))
pred = net.prediction(zero_input)
for digit in range(10):
    print(f"Digit {digit}: {pred[digit][0]*100:.2f}%")
```

#### Healthy Pattern:
No single digit should dominate. Ideally somewhat balanced.

#### Red Flags:
- ❌ **One digit >80%**: Strong bias
- ❌ **Same as random input test**: Model ignoring inputs

#### Why It Matters:
This tests the model's "default" behavior with no information. A healthy model shouldn't have extreme defaults.

---

## Practical Health Check Workflow

### Step 1: Quick Visual Check
```bash
uv run debug_model.py
```

Look for these sections:
1. **Weight Statistics** → Check mean absolute values
2. **Prediction Distribution** → Check for balance
3. **Recommendations** → Follow any warnings

### Step 2: Training Accuracy Check
Look at your training output:
- Is accuracy improving?
- Is it reaching >85% by epoch 10?
- Is train/test gap reasonable (<5%)?

### Step 3: Real-World Test
```bash
uv run main.py --predict 3.jpg
uv run main.py --predict 4.jpg
```
- Do different images get different predictions?
- Are predictions reasonable?

---

## Common Issues and Diagnosis

### Issue: "Model predicts everything as one digit"
**Diagnosis:**
- ❌ Prediction distribution test: One digit >50%
- ❌ Weight magnitudes: <0.01
- ❌ Training accuracy: Stuck at 11-15%

**Solution:**
- Fix weight initialization (use He init)
- Retrain from scratch

### Issue: "Low accuracy (<80%)"
**Diagnosis:**
- ✅ Prediction distribution: Balanced
- ✅ Weight magnitudes: Normal
- ❌ Training accuracy: Not improving

**Solution:**
- Increase epochs (try 20-30)
- Adjust learning rate (try 0.3, 1.0, 3.0)
- Increase hidden layer size

### Issue: "Training accuracy high, test accuracy low"
**Diagnosis:**
- ✅ Training accuracy: >95%
- ❌ Test accuracy: <85%
- ❌ Gap: >10%

**Solution:**
- Overfitting! Add regularization or reduce model complexity
- Use dropout or L2 regularization
- Get more training data

### Issue: "Accuracy oscillates wildly"
**Diagnosis:**
- Training accuracy jumps up and down
- No steady improvement

**Solution:**
- Learning rate too high
- Reduce eta (try 0.1, 0.3)

---

## Summary: Quick Health Checklist

Run through this checklist:

- [ ] **Weights**: Average magnitude 0.05-0.3? ✅/❌
- [ ] **Random predictions**: Balanced across all digits? ✅/❌
- [ ] **Training curve**: Steady improvement? ✅/❌
- [ ] **Final accuracy**: >85% on test set? ✅/❌
- [ ] **Real images**: Different predictions for different digits? ✅/❌

**If all ✅**: Your model is healthy! 🎉

**If any ❌**: Check the specific section above for diagnosis and solutions.

---

## Advanced: Understanding the Numbers

### Why sqrt(2/x) for He initialization?

**Mathematical reasoning:**
- During forward pass, variance of activations = variance of inputs × variance of weights × number of inputs
- We want variance to stay constant (≈1) through layers
- For ReLU activation: optimal variance = 2/n_in
- Standard deviation = sqrt(variance) = sqrt(2/n_in)

**Practical impact:**
- Too small (0.01): Variance shrinks exponentially → vanishing gradients
- Too large (1.0): Variance grows exponentially → exploding gradients
- Just right (sqrt(2/x)): Variance stays constant → stable learning

### Why test on random inputs?

**Purpose:**
- Random inputs have no pattern
- A well-trained model should be "confused" by random noise
- If it strongly prefers one class on noise, it's biased, not learned

**Interpretation:**
- Balanced predictions (10% each): Model has learned patterns, not biases
- Imbalanced (>50% one digit): Model is biased, ignoring actual input

---

## Tools Reference

### debug_model.py
Comprehensive automated health check. Run after training.

### Manual checks
```python
from network import NeuralNetwork
import numpy as np

net = NeuralNetwork.load("mnist_model.json")

# Check weight magnitude
print(np.mean(np.abs(net.weights[0])))  # Should be ~0.05

# Check prediction diversity
preds = [np.argmax(net.prediction(np.random.rand(784,1))) for _ in range(100)]
print(set(preds))  # Should see multiple different digits
```

### Training output
Watch the epoch-by-epoch accuracy. Should steadily improve.
