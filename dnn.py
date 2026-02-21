# Blueprint for a Deep Feedforward Neural Network (DNN)

# 1. Import necessary libraries (numpy for computation, optional: torch/tensorflow for larger projects)
import json
import numpy as np
from nnLayer import nnLayer

# 2. Define a DNN class
#    - Constructor should accept:
#        - layer_sizes: list of integers specifying the number of neurons in each layer
#        - activations: list of activation function names per layer
#    - Initialize weights and biases for each layer
#    - Store activation functions for each layer
class DeepNeuralNetwork:
    def __init__(self, layer_sizes, activations):
        assert len(layer_sizes) - 1 == len(activations), "Number of activations must be one less than number of layers."
        self.layers = []
        self.debug = []
        for i in range(len(layer_sizes) - 1):
            self.debug.append({
                'layer': i,
                'message': f'Initializing layer {i}: {layer_sizes[i]} -> {layer_sizes[i+1]} neurons, activation: {activations[i]}'
            })
            self.layers.append(nnLayer(layer_sizes[i], layer_sizes[i+1], activations[i]))

# 3. Implement the forward pass
#    - Accepts input data
#    - Iteratively computes activations for each layer
#    - Optionally applies dropout and batch normalization
#    - Returns final output and optionally intermediate activations for debugging
    def forward(self, x):
        for idx, layer in enumerate(self.layers):
            x = layer.forward(x, layer_idx=idx)
            layer.details(self.debug)
        return x

# 4. Implement the backward pass (backpropagation)
#    - Accepts loss gradient with respect to output
#    - Computes gradients for all weights and biases
#    - Optionally supports different optimizers (SGD, Adam, etc.)
#    - Updates parameters
    def backward(self, loss_grad, learning_rate):
        self.debug.append({
            'loss_grad_norm': round(float(np.linalg.norm(loss_grad)), 6),
            'learning_rate': learning_rate,
        })
        for idx, layer in enumerate(reversed(self.layers)):
            loss_grad = layer.backward(loss_grad, learning_rate, debug=self.debug, layer_idx=idx)
        return loss_grad

# 5. Add training loop method
#    - Accepts training data, labels, epochs, batch size, learning rate
#    - Handles batching, shuffling, and loss computation
#    - Tracks and prints/returns training progress (loss, accuracy)

# 6. Add evaluation method
#    - Accepts test/validation data
#    - Returns accuracy, loss, and other metrics

# 7. Add utility methods
#    - Save/load model parameters
#    - Print model summary (layer sizes, activations, parameter count)
#    - Debugging hooks (e.g., to log activations, gradients)
    def get_debug_json(self):
        return json.dumps(self.debug, indent=2)

# 8. (Optional) Support for advanced features
#    - Residual connections
#    - Custom activation functions
#    - Learning rate scheduling
#    - Early stopping
#    - Multi-GPU support (if using a deep learning framework)

# Example usage:
# dnn = DNN(layer_sizes=[784, 256, 128, 10], activations=['relu', 'relu', 'softmax'])
# dnn.train(X_train, y_train, epochs=20, batch_size=64, learning_rate=0.01)
# accuracy = dnn.evaluate(X_test, y_test)
# dnn.save('model_weights.npz')
