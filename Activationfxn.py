import numpy as np
import matplotlib.pyplot as plt

# ---------------- Activation Functions ----------------
def sigmoid(x):
    """Sigmoid function"""
    return 1.0 / (1.0 + np.exp(-x))

def tanh_fn(x):
    """Tanh function"""
    return np.sinh(x) / np.cosh(x)   # equivalent to np.tanh(x)

def relu_fn(x):
    """ReLU function"""
    return np.where(x > 0, x, 0)

def leaky_relu_fn(x, alpha=0.05):
    """Leaky ReLU function with slope alpha"""
    return np.where(x > 0, x, alpha * x)

def softmax_fn(values):
    """Softmax for a vector"""
    shift_vals = values - np.max(values)   # for numerical stability
    exp_vals = np.exp(shift_vals)
    return exp_vals / np.sum(exp_vals)

# ---------------- Visualization ----------------
x_vals = np.linspace(-10, 10, 500)

plt.figure(figsize=(12, 8))

# Sigmoid
plt.subplot(2, 2, 1)
plt.plot(x_vals, sigmoid(x_vals), color="blue")
plt.title("Sigmoid Function")
plt.axhline(0, color="black", linewidth=0.6)
plt.axvline(0, color="black", linewidth=0.6)
plt.grid(True, linestyle="--", alpha=0.6)

# Tanh
plt.subplot(2, 2, 2)
plt.plot(x_vals, tanh_fn(x_vals), color="green")
plt.title("Tanh Function")
plt.axhline(0, color="black", linewidth=0.6)
plt.axvline(0, color="black", linewidth=0.6)
plt.grid(True, linestyle="--", alpha=0.6)

# ReLU
plt.subplot(2, 2, 3)
plt.plot(x_vals, relu_fn(x_vals), color="red")
plt.title("ReLU Function")
plt.axhline(0, color="black", linewidth=0.6)
plt.axvline(0, color="black", linewidth=0.6)
plt.grid(True, linestyle="--", alpha=0.6)

# Leaky ReLU
plt.subplot(2, 2, 4)
plt.plot(x_vals, leaky_relu_fn(x_vals), color="purple")
plt.title("Leaky ReLU Function")
plt.axhline(0, color="black", linewidth=0.6)
plt.axvline(0, color="black", linewidth=0.6)
plt.grid(True, linestyle="--", alpha=0.6)

plt.tight_layout()
plt.show()

# ---------------- Softmax Visualization ----------------
vec = np.array([3.0, 1.0, 0.2])
softmax_probs = softmax_fn(vec)

plt.figure(figsize=(6,4))
plt.bar(range(len(vec)), softmax_probs, color=["blue","green","orange"])
plt.xticks(range(len(vec)), [f"z={val}" for val in vec])
plt.ylabel("Probability")
plt.title("Softmax Output (Probabilities)")
plt.show()
