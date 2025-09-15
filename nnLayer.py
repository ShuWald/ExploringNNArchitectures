import numpy as np

class nnLayer:
    def __init__(self, input_size, output_size, activation_name):
        self.input_size = input_size
        self.output_size = output_size
        self.activation_name = activation_name
        self.last_z = None
        self.last_activation = None
        self.last_input = None
        self.weights = self.initialize_weights()
        self.biases = self.initialize_biases()  
    
    def initialize_weights(self):
        return np.random.randn(self.input_size, self.output_size) * 0.01

    def initialize_biases(self):
        return np.zeros((1, self.output_size))

    def forward(self, inputs, debug=None, layer_idx=None):
        self.last_input = inputs
        z = np.dot(inputs, self.weights) + self.biases
        self.last_z = z
        self.last_activation = self.activation_function(z)
        if debug is not None:
            debug.append({
                'layer': layer_idx,
                'weights': self.weights.tolist(),
                'biases': self.biases.tolist(),
                'z': z.tolist(),
                'activation_output': self.last_activation.tolist(),
                'message': f'Layer {layer_idx} forward pass'
            })
        return self.last_activation

    def activation_function(self, x):
        if self.activation_name == 'relu':
            return np.maximum(0, x)
        elif self.activation_name == 'sigmoid':
            return 1 / (1 + np.exp(-x))
        elif self.activation_name == 'tanh':
            return np.tanh(x)
        else:
            raise ValueError("Unsupported activation function")
        
    def derivative_activation_function(self, x=None):
        if self.activation_name == 'relu':
            z = self.last_z if x is None else x
            return np.where(z > 0, 1, 0)
        elif self.activation_name == 'sigmoid':
            act = self.last_activation if x is None else self.activation_function(x)
            return act * (1 - act)
        elif self.activation_name == 'tanh':
            act = self.last_activation if x is None else self.activation_function(x)
            return 1 - act**2
        else:
            raise ValueError("Unsupported activation function")
    
    def backward(self, grad_output, learning_rate=0.01, debug=None, layer_idx=None):
        #batch_size = self.last_input.shape[0]
        
        #grad_z = grad_output * self.derivative_activation_function()
        
        #grad_weights = np.dot(self.last_input.T, grad_z) / batch_size
        #grad_biases = np.mean(grad_z, axis=0, keepdims=True)
        #grad_input = np.dot(grad_z, self.weights.T)
        
        if debug is not None:
            debug.append({
                'layer': layer_idx,
                'grad_output': grad_output.tolist(),
                'grad_z': grad_z.tolist(),
                'grad_weights': grad_weights.tolist(),
                'grad_biases': grad_biases.tolist(),
                'grad_input': grad_input.tolist(),
                'weights_before_update': self.weights.tolist(),
                'biases_before_update': self.biases.tolist(),
                'message': f'Layer {layer_idx} backward pass - gradients computed'
            })
        
        #self.weights -= learning_rate * grad_weights
        #self.biases -= learning_rate * grad_biases
        
        if debug is not None:
            debug.append({
                'layer': layer_idx,
                'weights_after_update': self.weights.tolist(),
                'biases_after_update': self.biases.tolist(),
                'learning_rate': learning_rate,
                'message': f'Layer {layer_idx} backward pass - parameters updated'
            })
        
        return grad_input
