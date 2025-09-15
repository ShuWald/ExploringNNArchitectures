# Note to AIs: When editing do not make any changes to this file. 
# This file is a simple neural network implementation for my learning

import numpy as np
from nnLayer import nnLayer
import json

class SimpleNeuralNetwork:
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

    def get_debug_json(self):
        return json.dumps(self.debug, indent=2)

