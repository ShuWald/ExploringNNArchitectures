import numpy as np
from nnLayer import nnLayer
import json

class ConvolutionalNeuralNetwork:
    def __init__(self, input_shape, conv_layers, fc_layers, activations):
        """
        Initialize CNN with convolutional and fully connected layers
        
        Args:
            input_shape: tuple (height, width, channels) of input images
            conv_layers: list of tuples (num_filters, filter_size, stride, padding)
            fc_layers: list of fully connected layer sizes
            activations: list of activation functions for each layer
        """
        self.input_shape = input_shape
        self.conv_layers = conv_layers
        self.fc_layers = fc_layers
        self.activations = activations
        self.layers = []
        self.debug = []
        
        # Initialize convolutional layers
        # TODO: Initialize convolutional layer parameters (filters, biases)
        # TODO: Calculate output shapes after each conv layer
        
        # Initialize fully connected layers
        # TODO: Calculate flattened size from last conv layer
        # TODO: Initialize FC layers using existing nnLayer class
        
        self.debug.append({
            'message': 'CNN initialized',
            'input_shape': input_shape,
            'conv_layers': conv_layers,
            'fc_layers': fc_layers
        })
    
    def conv2d(self, input_data, filters, bias, stride=1, padding=0):
        """
        Perform 2D convolution operation
        
        Args:
            input_data: Input feature maps
            filters: Convolutional filters/kernels
            bias: Bias terms for each filter
            stride: Stride for convolution
            padding: Padding around input
        
        Returns:
            Convolved feature maps
        """
        # TODO: Implement convolution operation
        # TODO: Handle padding (zero-padding, same, valid)
        # TODO: Apply stride
        # TODO: Add bias terms
        # TODO: Return convolved output
        pass
    
    def max_pool2d(self, input_data, pool_size=2, stride=2):
        """
        Perform 2D max pooling operation
        
        Args:
            input_data: Input feature maps
            pool_size: Size of pooling window
            stride: Stride for pooling
        
        Returns:
            Pooled feature maps
        """
        # TODO: Implement max pooling
        # TODO: Slide pooling window with given stride
        # TODO: Take maximum value in each window
        # TODO: Return pooled output
        pass
    
    def flatten(self, input_data):
        """
        Flatten multi-dimensional input to 1D for fully connected layers
        
        Args:
            input_data: Multi-dimensional input
        
        Returns:
            Flattened 1D array
        """
        # TODO: Reshape input to (batch_size, flattened_features)
        # TODO: Store original shape for backward pass
        pass
    
    def forward(self, x):
        """
        Forward pass through the CNN
        
        Args:
            x: Input batch of images
        
        Returns:
            Network output
        """
        self.debug.append({'input_shape': x.shape, 'message': 'Forward pass started'})
        
        # TODO: Pass through convolutional layers
        # TODO: Apply convolution, activation, and pooling for each conv layer
        # TODO: Store intermediate results for backward pass
        
        # TODO: Flatten the output from conv layers
        
        # TODO: Pass through fully connected layers
        # TODO: Use existing nnLayer forward method for FC layers
        
        self.debug.append({'output_shape': x.shape, 'message': 'Forward pass completed'})
        return x
    
    def backward(self, grad_output, learning_rate=0.01):
        """
        Backward pass through the CNN (backpropagation)
        
        Args:
            grad_output: Gradient of loss with respect to output
            learning_rate: Learning rate for parameter updates
        
        Returns:
            Gradient with respect to input
        """
        # TODO: Backward pass through fully connected layers
        # TODO: Use existing nnLayer backward method for FC layers
        
        # TODO: Reshape gradients for convolutional layers
        
        # TODO: Backward pass through pooling layers
        # TODO: Propagate gradients through max pooling
        
        # TODO: Backward pass through convolutional layers
        # TODO: Compute gradients for filters and biases
        # TODO: Update filter weights and biases
        
        # TODO: Return gradient with respect to input
        pass
    
    def conv_backward(self, grad_output, input_data, filters, stride=1, padding=0):
        """
        Backward pass for convolution layer
        
        Args:
            grad_output: Gradient from next layer
            input_data: Input to this conv layer
            filters: Filters used in forward pass
            stride: Stride used in forward pass
            padding: Padding used in forward pass
        
        Returns:
            Gradients for filters, bias, and input
        """
        # TODO: Compute gradient with respect to filters
        # TODO: Compute gradient with respect to bias
        # TODO: Compute gradient with respect to input
        pass
    
    def get_debug_json(self):
        """Return debug information as JSON string"""
        return json.dumps(self.debug, indent=2)