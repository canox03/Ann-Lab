import numpy as np
import matplotlib.pyplot as plt

class Perceptron:
    def __init__(self, input_size, lr=0.1, epochs=100):
        self.weights = np.zeros(input_size + 1)
        self.lr = lr
        self.epochs = epochs
    
    def activation_fn(self, x):
        return 1 if x >= 0 else 0
    
    def predict(self, inputs):
        summation = np.dot(inputs, self.weights[1:]) + self.weights[0]
        return self.activation_fn(summation)
    
    def train(self, training_inputs, labels):
        for _ in range(self.epochs):
            for inputs, label in zip(training_inputs, labels):
                prediction = self.predict(inputs)
                self.weights[1:] += self.lr * (label - prediction) * inputs
                self.weights[0] += self.lr * (label - prediction)

# Generate random data points
np.random.seed(0)
class1 = np.random.randn(50, 2) + np.array([2, 2])
class2 = np.random.randn(50, 2) + np.array([-2, -2])

# Plot the data points
plt.scatter(class1[:,0], class1[:,1], color='b', marker='o', label='Class 1')
plt.scatter(class2[:,0], class2[:,1], color='r', marker='x', label='Class 2')

# Prepare training data and labels
training_inputs = np.vstack([class1, class2])
labels = np.hstack([np.zeros(len(class1)), np.ones(len(class2))])

# Create and train the perceptron
perceptron = Perceptron(input_size=2)
perceptron.train(training_inputs, labels)

# Plot the decision boundary
x_min, x_max = plt.xlim()
y_min, y_max = plt.ylim()
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                     np.arange(y_min, y_max, 0.1))
Z = np.array([perceptron.predict(np.array([x, y])) for x, y in np.c_[xx.ravel(), yy.ravel()]])
Z = Z.reshape(xx.shape)
plt.contour(xx, yy, Z, colors='k', levels=[-1, 0, 1], alpha=0.5,
            linestyles=['--', '-', '--'])

plt.title('Perceptron Decision Regions')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.show()
