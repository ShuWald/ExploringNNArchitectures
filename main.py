import numpy as np
from snn import SimpleNeuralNetwork

if __name__ == "__main__":
    np.random.seed(42)
    # Define a simple network: 2 inputs, 1 hidden layer (3 neurons), 1 output
    nn = SimpleNeuralNetwork(layer_sizes=[2, 3, 1], activations=['relu', 'sigmoid'])
    test_input = np.array([[0.5, -0.2], [1.0, 0.3]])
    print("\nRunning test input through the neural network:")
    output = nn.forward(test_input)
    print(f"\nNetwork output for test input:\n{output}")
    print("\nDebug trace:")
    print(nn.get_debug_json())
