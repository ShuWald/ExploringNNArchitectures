import numpy as np
from snn import SimpleNeuralNetwork
import os

def write_debug_to_file(debug_json, filename="debug_output.txt"):
    debug_folder = "Debug"
    os.makedirs(debug_folder, exist_ok=True)
    filepath = os.path.join(debug_folder, filename)
    with open(filepath, 'w') as f:
        f.write(debug_json)
    print(f"Debug information written to {filepath}")

if __name__ == "__main__":
    np.random.seed(42)
    nn = SimpleNeuralNetwork(layer_sizes=[2, 3, 1], activations=['relu', 'sigmoid'])
    test_input = np.array([[0.5, -0.2], [1.0, 0.3]])
    
    print("\nRunning test input through the neural network:")
    output = nn.forward(test_input)
    print(f"\nNetwork output for test input:\n{output}")
    '''
    print("\nTesting backward propagation:")
    grad_output = np.array([[1.0], [1.0]])
    nn.backward(grad_output, learning_rate=0.01)
    '''
    write_debug_to_file(nn.get_debug_json(), "simplenn_debug.txt")
