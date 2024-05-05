import numpy as np

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.1):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        
        # Initialize weights and biases
        self.weights_input_hidden = np.random.randn(self.input_size, self.hidden_size)
        self.bias_hidden = np.zeros((1, self.hidden_size))
        self.weights_hidden_output = np.random.randn(self.hidden_size, self.output_size)
        self.bias_output = np.zeros((1, self.output_size))
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def sigmoid_derivative(self, x):
        return x * (1 - x)
    
    def forward_propagation(self, inputs):
        # Hidden layer calculation
        self.hidden_output = self.sigmoid(np.dot(inputs, self.weights_input_hidden) + self.bias_hidden)
        
        # Output layer calculation
        self.output = self.sigmoid(np.dot(self.hidden_output, self.weights_hidden_output) + self.bias_output)
        return self.output
    
    def backward_propagation(self, inputs, targets):
        # Output layer error
        output_error = targets - self.output
        output_delta = output_error * self.sigmoid_derivative(self.output)
        
        # Hidden layer error
        hidden_error = np.dot(output_delta, self.weights_hidden_output.T)
        hidden_delta = hidden_error * self.sigmoid_derivative(self.hidden_output)
        
        # Update weights and biases
        self.weights_hidden_output += np.dot(self.hidden_output.T, output_delta) * self.learning_rate
        self.bias_output += np.sum(output_delta, axis=0, keepdims=True) * self.learning_rate
        self.weights_input_hidden += np.dot(inputs.T, hidden_delta) * self.learning_rate
        self.bias_hidden += np.sum(hidden_delta, axis=0, keepdims=True) * self.learning_rate
    
    def train(self, inputs, targets, epochs):
        for epoch in range(epochs):
            # Forward propagation
            output = self.forward_propagation(inputs)
            
            # Backward propagation
            self.backward_propagation(inputs, targets)
            
            # Print loss every 100 epochs
            if (epoch+1) % 100 == 0:
                loss = np.mean(np.square(targets - output))
                print(f"Epoch {epoch+1}/{epochs}, Loss: {loss:.4f}")

# Example usage
# Define training data (XOR problem)
inputs = np.array([[0, 0],
                   [0, 1],
                   [1, 0],
                   [1, 1]])

targets = np.array([[0],
                    [1],
                    [1],
                    [0]])

# Create a neural network with 2 input neurons, 2 hidden neurons, and 1 output neuron
nn = NeuralNetwork(input_size=2, hidden_size=2, output_size=1, learning_rate=0.1)

# Train the neural network for 1000 epochs
nn.train(inputs, targets, epochs=1000)
