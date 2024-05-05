import numpy as np
import matplotlib.pyplot as plt

# Sigmoid function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Tanh function
def tanh(x):
    return np.tanh(x)

# ReLU function
def relu(x):
    return np.maximum(0, x)

# Leaky ReLU function
def leaky_relu(x, alpha=0.01):
    return np.maximum(alpha*x, x)

# Plotting
x = np.linspace(-5, 5, 100)
plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.plot(x, sigmoid(x))
plt.title('Sigmoid Function')

plt.subplot(2, 2, 2)
plt.plot(x, tanh(x))
plt.title('Tanh Function')

plt.subplot(2, 2, 3)
plt.plot(x, relu(x))
plt.title('ReLU Function')

plt.subplot(2, 2, 4)
plt.plot(x, leaky_relu(x))
plt.title('Leaky ReLU Function')

plt.tight_layout()
plt.show()