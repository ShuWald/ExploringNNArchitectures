import numpy as np

class nnLayer:
    def __init__(self, input_size, output_size, activation_name):
        self.input_size = input_size
        self.output_size = output_size
        self.activation_name = activation_name
        self.last_z = None
        self.last_activation = None
        self.last_input = None

        #Will explore different weights/biases initialization techniques later
        self.weights = self.initialize_weights()
        self.biases = self.initialize_biases()  

    def initialize_weights(self):
        # He initialization for relu, Xavier for sigmoid/tanh
        # Scales weights so activation variance stays ~1 across layers regardless of fan_in
        if self.activation_name == 'relu':
            # ReLU kills ~half the signal, so scale up by sqrt(2) to compensate
            return np.random.randn(self.input_size, self.output_size) * np.sqrt(2.0 / self.input_size)
        else:
            # sigmoid/tanh: no signal killed, scale by sqrt(1/fan_in)
            return np.random.randn(self.input_size, self.output_size) * np.sqrt(1.0 / self.input_size)
    def initialize_biases(self):
        return np.zeros((1, self.output_size))

    def forward(self, inputs, layer_idx=None):
        z = np.dot(inputs, self.weights) + self.biases
        #Remember this layer's latest variables(for backpropagation)
        self.last_input = inputs
        self.last_z = z
        self.last_activation = self.activation_function(z)           
        return self.last_activation
    
    def details(self, debug):
        # Log scalar summaries only — avoids dumping full matrices every epoch
        debug.append({
            'weights_shape': list(self.weights.shape),
            'weights_mean': round(float(np.mean(self.weights)), 6),
            'weights_std': round(float(np.std(self.weights)), 6),
            'biases_mean': round(float(np.mean(self.biases)), 6),
            'activation_mean': round(float(np.mean(self.last_activation)), 6),
            'activation_std': round(float(np.std(self.last_activation)), 6),
        })

    #Some common activation functions
    def activation_function(self, x):
        if self.activation_name == 'relu':
            return np.maximum(0, x)
        elif self.activation_name == 'sigmoid':
            return 1 / (1 + np.exp(-x))
        elif self.activation_name == 'tanh':
            return np.tanh(x)
        elif self.activation_name == 'linear':
            return x  # No transformation — used for regression output layers
        else:
            raise ValueError("Unsupported activation function")

    #Takes the derivative of the activation function    
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
        elif self.activation_name == 'linear':
            return np.ones_like(self.last_z if x is None else x)  # Derivative is 1 everywhere
        else:
            raise ValueError("Unsupported activation function")
    
    def backward(self, grad_output, learning_rate=0.01, debug=None, layer_idx=None):
        batch_size = self.last_input.shape[0]
        grad_z = grad_output * self.derivative_activation_function()
        
        # [input_size x batch_size] @ [batch_size x output_size] = [input_size x output_size]
        grad_weights = np.dot(self.last_input.T, grad_z) / batch_size
        # [batch_size x output_size] -> mean along batch_size axis -> [1 x output_size]
        grad_biases = np.mean(grad_z, axis=0, keepdims=True)
        # [batch_size x output_size] @ [input_size x output_size].Transpose = [batch_size x input_size]
        grad_input = np.dot(grad_z, self.weights.T)
        
        self.weights -= learning_rate * grad_weights
        self.biases -= learning_rate * grad_biases

        # Single compact entry: scalar summaries only
        debug.append({
            'layer': layer_idx,
            'grad_output_norm': round(float(np.linalg.norm(grad_output)), 6),
            'grad_z_norm': round(float(np.linalg.norm(grad_z)), 6),
            'grad_weights_norm': round(float(np.linalg.norm(grad_weights)), 6),
            'grad_biases_mean': round(float(np.mean(grad_biases)), 6),
            'weights_mean': round(float(np.mean(self.weights)), 6),
            'weights_std': round(float(np.std(self.weights)), 6),
        })
        
        return grad_input
