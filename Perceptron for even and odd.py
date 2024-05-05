import numpy as np

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

# Training data - ASCII representation of numbers 0 to 9
training_inputs = np.array([
    [0,0,0,0,0,0,0,0,0,0,1], # ASCII representation of '0'
    [1,0,0,0,0,0,0,0,0,0,1], # ASCII representation of '1'
    [0,1,0,0,0,0,0,0,0,0,1], # ASCII representation of '2'
    [1,1,0,0,0,0,0,0,0,0,1], # ASCII representation of '3'
    [0,0,1,0,0,0,0,0,0,0,1], # ASCII representation of '4'
    [1,0,1,0,0,0,0,0,0,0,1], # ASCII representation of '5'
    [0,1,1,0,0,0,0,0,0,0,1], # ASCII representation of '6'
    [1,1,1,0,0,0,0,0,0,0,1], # ASCII representation of '7'
    [0,0,0,1,0,0,0,0,0,0,1], # ASCII representation of '8'
    [1,0,0,1,0,0,0,0,0,0,1], # ASCII representation of '9'
])

# Labels: 0 for even numbers, 1 for odd numbers
labels = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])

# Create and train the perceptron
perceptron = Perceptron(input_size=10)
perceptron.train(training_inputs, labels)

# Test the perceptron
test_inputs = np.array([
    [0,0,0,0,0,0,0,0,0,0,1], # ASCII representation of '0'
    [0,0,1,1,0,0,0,0,0,0,1], # ASCII representation of '4'
    [1,0,0,1,0,0,0,0,0,0,1], # ASCII representation of '9'
])

for inputs in test_inputs:
    prediction = perceptron.predict(inputs)
    number = chr(int(''.join(map(str, inputs[:-1])), 2))
    print(f"Prediction for '{number}' is {'Odd' if prediction == 1 else 'Even'}")