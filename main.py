import numpy as np
from dnn import DeepNeuralNetwork
from snn import SimpleNeuralNetwork
import os

def write_debug_to_file(debug_json, filename="debug_output.txt"):
    debug_folder = "Debug"
    os.makedirs(debug_folder, exist_ok=True)
    filepath = os.path.join(debug_folder, filename)
    with open(filepath, 'w') as f:
        f.write(debug_json)
    print(f"Debug information written to {filepath}")

def mse_loss(y_pred, y_true):
    return np.mean((y_pred - y_true)**2)

def mse_loss_derivative(y_pred, y_true):
    return 2 * (y_pred - y_true) / y_true.shape[0]

def trainnn(model, train_input, train_output, lr=0.01, debug_filename="debug_output.txt", epochs=1, loss_function="mse", batch_size=None):
    print("\nTraining the neural network...")
    n = train_input.shape[0]
    effective_batch = batch_size if batch_size is not None else n
    for epoch in range(epochs):
        # Shuffle data
        indices = np.random.permutation(n)
        x_shuffled = train_input[indices]
        y_shuffled = train_output[indices]

        epoch_loss = 0
        num_batches = 0
        for start in range(0, n, effective_batch):
            x_batch = x_shuffled[start:start + effective_batch]
            y_batch = y_shuffled[start:start + effective_batch]

            output = model.forward(x_batch)
            epoch_loss += mse_loss(output, y_batch)
            num_batches += 1

            if loss_function == "none":
                # SE gradient: dL/dy_pred = (y_pred - y_true)
                grad_output = (output - y_batch)
            else:
                # MSE gradient: dL/dy_pred = 2*(y_pred-y_true)/N
                grad_output = mse_loss_derivative(output, y_batch)
            model.backward(grad_output, learning_rate=lr)

        if epoch % max(1, epochs // 10) == 0:
            print(f"Epoch {epoch+1}/{epochs} - Avg Loss: {epoch_loss/num_batches:.6f}")

    #model.debug.clear()
    model.forward(train_input)  # Final forward pass on full data for debug snapshot
    write_debug_to_file(model.get_debug_json(), debug_filename)

def testnn(model, test_input, test_output):
    model_output = model.forward(test_input)
    print(f"\nModel output for test input:\n{model_output}")
    #print(f"\nExpected output for test input:\n{test_output}")
    accuracy = np.mean(np.isclose(model_output, test_output, atol=0.1)) * 100
    print(f"\nModel accuracy: {accuracy:.2f}%")

if __name__ == "__main__":
    np.random.seed(42)
    # Single hidden ReLU layer avoids stacked dying-ReLU killing 75% of neurons
    snn = SimpleNeuralNetwork(layer_sizes=[2, 8, 1], activations=['relu', 'linear'])  
    dnn = DeepNeuralNetwork(layer_sizes=[2, 8, 6, 1], activations=['relu', 'relu', 'linear']) 

    data_input = np.random.uniform(0, 0.5, (500, 2))  # 500 samples, 2 features each
    # Pattern: y = sigmoid(x1 + x2)
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    data_output = (data_input[:, 0] + data_input[:, 1])  
    data_output = data_output.reshape(-1, 1)  # Reshape to match the output shape

    train_input = data_input[:475]
    train_output = data_output[:475]
    test_input = data_input[475:500]
    test_output = data_output[475:500]

    trainnn(snn, train_input, train_output, 0.1, debug_filename="snn_debug.txt", epochs=100, batch_size=32)
    testnn(snn, test_input, test_output)

    trainnn(dnn, train_input, train_output, 0.1, debug_filename="dnn_debug.txt", epochs=100, batch_size=32)
    testnn(dnn, test_input, test_output)

'''
    snn2 = SimpleNeuralNetwork(layer_sizes=[2, 3, 1], activations=['relu', 'linear'])  
    dnn2 = DeepNeuralNetwork(layer_sizes=[2, 4, 1], activations=['relu', 'linear']) 

    trainnn(snn2, train_input, train_output, 0.5, debug_filename="snn2_debug.txt", loss_function="none", epochs=25, batch_size=32)
    testnn(snn2, test_input, test_output)

    trainnn(dnn2, train_input, train_output, 0.5, debug_filename="dnn2_debug.txt", loss_function="none", epochs=25, batch_size=32)
    testnn(dnn2, test_input, test_output)
'''
