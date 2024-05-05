import numpy as np

class McCullochPittsNeuron:
    def __init__(self, weights, threshold):
        self.weights = weights
        self.threshold = threshold
    
    def activate(self, inputs):
        weighted_sum = np.dot(inputs, self.weights)
        if weighted_sum >= self.threshold:
            return 1
        else:
            return 0

def ANDNOT(x1, x2):
    # Weights and threshold for ANDNOT function
    weights = [-1, -1]  # Using negative weights
    threshold = -0.5
    
    # Create ANDNOT neuron
    neuron = McCullochPittsNeuron(weights, threshold)
    
    # Input for ANDNOT function
    inputs = np.array([x1, x2])
    
    # Activate neuron
    return neuron.activate(inputs)

# Test cases
print("ANDNOT(0, 0) =", ANDNOT(0, 0))
print("ANDNOT(0, 1) =", ANDNOT(0, 1))
print("ANDNOT(1, 0) =", ANDNOT(1, 0))
print("ANDNOT(1, 1) =", ANDNOT(1, 1))