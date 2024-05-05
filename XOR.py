import numpy as np

class NeuralNetwork:
    def __init__(self):
        # Initialize weights and biases for the hidden layer
        self.weights_hidden = np.random.rand(2, 2)  # 2 input neurons, 2 hidden neurons
        self.bias_hidden = np.random.rand(1, 2)     # 2 biases for 2 hidden neurons

        # Initialize weights and bias for the output layer
        self.weights_output = np.random.rand(2, 1)  # 2 hidden neurons, 1 output neuron
        self.bias_output = np.random.rand(1, 1)     # 1 bias for the output neuron

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def forward_propagation(self, X):
        # Forward propagation through hidden layer
        self.hidden_output = self.sigmoid(np.dot(X, self.weights_hidden) + self.bias_hidden)

        # Forward propagation through output layer
        self.output = self.sigmoid(np.dot(self.hidden_output, self.weights_output) + self.bias_output)

    def backward_propagation(self, X, y):
        # Compute the error and delta for the output layer
        output_error = y - self.output
        output_delta = output_error * self.sigmoid_derivative(self.output)

        # Compute the error and delta for the hidden layer
        hidden_error = np.dot(output_delta, self.weights_output.T)
        hidden_delta = hidden_error * self.sigmoid_derivative(self.hidden_output)

        # Update weights and biases for the output layer
        self.weights_output += np.dot(self.hidden_output.T, output_delta)
        self.bias_output += np.sum(output_delta, axis=0, keepdims=True)

        # Update weights and biases for the hidden layer
        self.weights_hidden += np.dot(X.T, hidden_delta)
        self.bias_hidden += np.sum(hidden_delta, axis=0, keepdims=True)

    def train(self, X, y, epochs):
        for epoch in range(epochs):
            # Forward propagation
            self.forward_propagation(X)

            # Backward propagation
            self.backward_propagation(X, y)

            # Print the loss every 100 epochs
            if (epoch + 1) % 100 == 0:
                loss = np.mean(np.square(y - self.output))
                print(f"Epoch {epoch+1}/{epochs}, Loss: {loss:.4f}")

    def predict(self, X):
        self.forward_propagation(X)
        return self.output

# Define the XOR inputs and outputs
X = np.array([[0, 0],
              [0, 1],
              [1, 0],
              [1, 1]])

y = np.array([[0],
              [1],
              [1],
              [0]])

# Create and train the neural network
nn = NeuralNetwork()
nn.train(X, y, epochs=1000)

# Test the trained neural network
print("\nPredictions after training:")
for i in range(len(X)):
    prediction = nn.predict(X[i])
    print(f"Input: {X[i]}, Predicted Output: {prediction[0]}")
