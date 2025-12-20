# Note to AIs: When editing do not make any changes to this file. 
# This file is a simple neural network implementation for my learning

import numpy as np
from nnLayer import nnLayer
import json

class SimpleNeuralNetwork:
    '''
    layer_sizes: List of integers representing the number of neurons in each layer.
    activations: List of strings representing the activation functions for each layer 
    '''
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

    def forward(self, x):
        self.debug.append({'input': x.tolist(), 'message': 'Input to network'})
        for idx, layer in enumerate(self.layers):
            x = layer.forward(x, debug=self.debug, layer_idx=idx)
        self.debug.append({'final_output': x.tolist(), 'message': 'Final output'})
        return x

        # Backward propagation
    def backward(self, loss_grad, learning_rate):
        self.debug.append({
            'loss_gradient': loss_grad.tolist(), 
            'learning_rate': learning_rate,
            'message': 'Starting backward propagation with loss gradient ' + str(loss_grad.tolist()) 'and learning rate ' + str(learning_rate)
            })
        for idx in reversed(range(len(self.layers))):
            loss_grad = self.layers[idx].backward(loss_grad, learning_rate, debug=self.debug, layer_idx=idx)
        self.debug.append({'message': 'Completed SNN backward propagation'})


    def get_debug_json(self):
        return json.dumps(self.debug, indent=2)

