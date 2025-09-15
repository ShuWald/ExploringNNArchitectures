import numpy as np
from nnLayer import nnLayer
import json

class RecurrentNeuralNetworks:
    def __init__(self, input_size, hidden_size, output_size, sequence_length, rnn_type='vanilla'):
        """
        Initialize RNN architecture
        
        Args:
            input_size: Size of input features at each time step
            hidden_size: Size of hidden state
            output_size: Size of output
            sequence_length: Length of input sequences
            rnn_type: Type of RNN ('vanilla', 'lstm', 'gru')
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.sequence_length = sequence_length
        self.rnn_type = rnn_type
        self.debug = []
        
        # Initialize weights and biases
        # TODO: Initialize input-to-hidden weights (Wxh)
        # TODO: Initialize hidden-to-hidden weights (Whh)
        # TODO: Initialize hidden-to-output weights (Why)
        # TODO: Initialize bias terms (bh, by)
        
        # For LSTM: initialize forget, input, output gates
        # TODO: Initialize LSTM gate weights and biases if rnn_type == 'lstm'
        
        # For GRU: initialize update and reset gates
        # TODO: Initialize GRU gate weights and biases if rnn_type == 'gru'
        
        # Initialize output layer
        # TODO: Use nnLayer for final output transformation if needed
        
        self.debug.append({
            'message': 'RNN initialized',
            'input_size': input_size,
            'hidden_size': hidden_size,
            'output_size': output_size,
            'rnn_type': rnn_type
        })
    
    def init_hidden_state(self, batch_size):
        """
        Initialize hidden state for a new sequence
        
        Args:
            batch_size: Number of sequences in batch
        
        Returns:
            Initial hidden state
        """
        # TODO: Initialize hidden state (usually zeros)
        # TODO: For LSTM, also initialize cell state
        pass
    
    def vanilla_rnn_step(self, x_t, h_prev):
        """
        Single time step of vanilla RNN
        
        Args:
            x_t: Input at current time step
            h_prev: Hidden state from previous time step
        
        Returns:
            New hidden state
        """
        # TODO: Compute h_t = tanh(Wxh @ x_t + Whh @ h_prev + bh)
        # TODO: Apply activation function (tanh)
        # TODO: Return new hidden state
        pass
    
    def lstm_step(self, x_t, h_prev, c_prev):
        """
        Single time step of LSTM
        
        Args:
            x_t: Input at current time step
            h_prev: Hidden state from previous time step
            c_prev: Cell state from previous time step
        
        Returns:
            New hidden state and cell state
        """
        # TODO: Compute forget gate: f_t = sigmoid(Wf @ [h_prev, x_t] + bf)
        # TODO: Compute input gate: i_t = sigmoid(Wi @ [h_prev, x_t] + bi)
        # TODO: Compute candidate values: C_tilde = tanh(WC @ [h_prev, x_t] + bC)
        # TODO: Compute output gate: o_t = sigmoid(Wo @ [h_prev, x_t] + bo)
        
        # TODO: Update cell state: C_t = f_t * C_prev + i_t * C_tilde
        # TODO: Update hidden state: h_t = o_t * tanh(C_t)
        
        # TODO: Return new hidden state and cell state
        pass
    
    def gru_step(self, x_t, h_prev):
        """
        Single time step of GRU
        
        Args:
            x_t: Input at current time step
            h_prev: Hidden state from previous time step
        
        Returns:
            New hidden state
        """
        # TODO: Compute update gate: z_t = sigmoid(Wz @ [h_prev, x_t] + bz)
        # TODO: Compute reset gate: r_t = sigmoid(Wr @ [h_prev, x_t] + br)
        # TODO: Compute candidate hidden state: h_tilde = tanh(Wh @ [r_t * h_prev, x_t] + bh)
        # TODO: Compute new hidden state: h_t = (1 - z_t) * h_prev + z_t * h_tilde
        
        # TODO: Return new hidden state
        pass
    
    def forward(self, x):
        """
        Forward pass through the RNN
        
        Args:
            x: Input sequences (batch_size, sequence_length, input_size)
        
        Returns:
            Network output
        """
        batch_size = x.shape[0]
        self.debug.append({
            'input_shape': x.shape,
            'message': 'RNN forward pass started'
        })
        
        # TODO: Initialize hidden state (and cell state for LSTM)
        hidden_states = []
        outputs = []
        
        # TODO: Process each time step
        for t in range(self.sequence_length):
            # TODO: Extract input at time step t
            # TODO: Apply appropriate RNN step (vanilla/LSTM/GRU)
            # TODO: Store hidden states and outputs
            # TODO: Update debug information
            pass
        
        # TODO: Compute final output from last hidden state or all hidden states
        # TODO: Apply output layer transformation if needed
        
        self.debug.append({
            'output_shape': "output.shape",
            'message': 'RNN forward pass completed'
        })
        
        # TODO: Return final output
        pass
    
    def backward(self, grad_output, learning_rate=0.01):
        """
        Backward pass through RNN (Backpropagation Through Time - BPTT)
        
        Args:
            grad_output: Gradient of loss with respect to output
            learning_rate: Learning rate for parameter updates
        
        Returns:
            Gradient with respect to input
        """
        # TODO: Initialize gradients for all parameters
        
        # TODO: Backward pass through output layer
        
        # TODO: Backpropagation through time
        # TODO: Start from last time step and go backwards
        for t in reversed(range(self.sequence_length)):
            # TODO: Compute gradients for current time step
            # TODO: Accumulate gradients for weights and biases
            # TODO: Propagate gradients to previous time step
            pass
        
        # TODO: Update parameters using computed gradients
        # TODO: Clip gradients to prevent exploding gradients
        
        # TODO: Return gradient with respect to input
        pass
    
    def vanilla_rnn_backward_step(self, grad_h, x_t, h_prev, h_curr):
        """
        Backward pass for single vanilla RNN step
        
        Args:
            grad_h: Gradient with respect to current hidden state
            x_t: Input at current time step
            h_prev: Previous hidden state
            h_curr: Current hidden state
        
        Returns:
            Gradients for weights, biases, and previous hidden state
        """
        # TODO: Compute gradient with respect to weights (Wxh, Whh)
        # TODO: Compute gradient with respect to biases
        # TODO: Compute gradient with respect to previous hidden state
        # TODO: Compute gradient with respect to input
        pass
    
    def lstm_backward_step(self, grad_h, grad_c, x_t, h_prev, c_prev, gates):
        """
        Backward pass for single LSTM step
        
        Args:
            grad_h: Gradient with respect to hidden state
            grad_c: Gradient with respect to cell state
            x_t: Input at time step
            h_prev: Previous hidden state
            c_prev: Previous cell state
            gates: Dictionary containing gate values from forward pass
        
        Returns:
            Gradients for all LSTM parameters
        """
        # TODO: Compute gradients through output gate
        # TODO: Compute gradients through cell state
        # TODO: Compute gradients through forget gate
        # TODO: Compute gradients through input gate
        # TODO: Compute gradients through candidate values
        # TODO: Compute gradients for all weight matrices and biases
        pass
    
    def clip_gradients(self, gradients, max_norm=5.0):
        """
        Clip gradients to prevent exploding gradients
        
        Args:
            gradients: Dictionary of gradients
            max_norm: Maximum allowed gradient norm
        
        Returns:
            Clipped gradients
        """
        # TODO: Compute total gradient norm
        # TODO: Scale gradients if norm exceeds max_norm
        # TODO: Return clipped gradients
        pass
    
    def get_debug_json(self):
        """Return debug information as JSON string"""
        return json.dumps(self.debug, indent=2)